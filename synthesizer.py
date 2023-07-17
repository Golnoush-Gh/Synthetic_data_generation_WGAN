import json

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from keras.optimizers import Adam, RMSprop
from numpy.random import randn
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from keras.models import load_model
import keras.backend
from keras.utils import to_ordinal
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.initializers import RandomNormal, RandomUniform
from table_evaluator import load_data, TableEvaluator

# make it deterministic
# tf.random.set_seed(42)
# np.random.seed(42)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, units, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)
        self.bias = self.add_weight('bias',
                                    shape=[units],
                                    initializer='zeros',
                                    trainable=True)

    def call(self, x):
        return x + self.bias


# clip model weights to a given hypercube
class ClipConstraint(tf.keras.constraints.Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return keras.backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


class Synthesizer:
    def __init__(self, noise_dimensions: int, batch_size: int, input_dimensions: int, n_categories: int,
                 output_dimensions: int, n_numerical_features: int, ordinal_feature: str, continuous_feature: str):
        self.effort_bands = None
        self.specials_cdf = None
        self.generator = None
        self.ordinal_feature = ordinal_feature
        self.continuous_feature = continuous_feature
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.noise_dimensions = noise_dimensions
        self.n_categories = n_categories
        self.n_numerical_features = n_numerical_features
        self.cat_boundaries = n_categories - 1
        self.batch_size = batch_size
        self.define_discriminator()
        self.define_generator()
        self.define_gan()
        self.data_categories = None
        self.scaler_income = None
        self.scaler_effort = None
        self.g_history = []
        self.d_history = []

    @staticmethod
    def from_data_spec(data_specs: Dict):
        return Synthesizer(
            noise_dimensions=data_specs.get('noise_dimensions', 2),
            batch_size=data_specs.get('batch_size', 256),
            input_dimensions=data_specs.get('input_dimensions', 2),
            output_dimensions=data_specs.get('output_dimensions', 2),
            n_categories=data_specs.get('n_categories', 10),
            n_numerical_features=data_specs.get('n_numerical_features', 1),
            ordinal_feature=data_specs['ordinal_feature'],
            continuous_feature=data_specs['continuous_feature'],
        )

    @classmethod
    def from_json(cls, spec_file: str):
        with open(spec_file) as f:
            return cls.from_data_spec(json.load(f))

    def learn(self, data: pd.DataFrame, num_iterations: int) -> None:
        """
        Fit the model to the data
        :param data: DataFrame input data
        :param num_iterations: number of iterations
        """
        scaled_data = self.preprocess(data)
        self.fit_to_data(scaled_data, num_iterations)

    def generate(self, num_rows: int) -> pd.DataFrame:
        """
        Generate simulated data
        :param num_rows: number of rows to generate
        :return: returns the genrated dataframe
        """
        X, y = self.generate_simulated_samples(num_rows)
        data_fake = self.postprocess(X)
        return data_fake

    def define_generator(self) -> None:
        """
        Generator to produce fake data from input random noise
        """
        inputs = tf.keras.Input(shape=self.noise_dimensions, name="inputs")
        initializer = RandomNormal(mean=0.0, stddev=0.02)
        layer_1 = Dense(50, activation='LeakyReLU', kernel_initializer=initializer,
                        bias_initializer='zeros', name='gen_layer1')(inputs)
        layer_1_norm = BatchNormalization(momentum=0.9, name='layer_1_norm')(layer_1)
        layer_2 = Dense(50, activation='LeakyReLU', name='gen_layer2')(layer_1_norm)
        layer_2_norm = BatchNormalization(momentum=0.9, name='layer_2_norm')(layer_2)
        gen_layer3 = Dense(1, activation='linear', name='gen_layer3')(layer_2_norm)

        gen_layer_cat_dense = Dense(1, activation='linear', name='gen_layer_cat_dense')(layer_2_norm)

        gen_layer_cat_bias = BiasLayer(self.cat_boundaries)(gen_layer_cat_dense)
        gen_layer_cat_activate = tf.keras.layers.Activation(activation='sigmoid')(gen_layer_cat_bias)

        concatenated = tf.keras.layers.Concatenate()([gen_layer3, gen_layer_cat_activate])

        self.generator = tf.keras.Model(inputs=inputs, outputs=concatenated, name='Generator')

    def define_discriminator(self) -> None:
        """
        Creat discriminator, it is used to
        classify examples as real (from the domain)
        or fake (generated).
        """
        initializer = RandomNormal(mean=0.0, stddev=0.02)
        const = ClipConstraint(0.01)
        model = Sequential(name='discriminator')
        # this was to stop the discriminator picking one answer
        model.add(tf.keras.layers.GaussianNoise(0.02, seed=42, name='discrim_noise'))
        model.add(Dense(25, activation='LeakyReLU', kernel_initializer=initializer,
                        input_dim=(self.n_numerical_features + self.cat_boundaries),
                        name='discrim_layer1', kernel_constraint=const))
        model.add(Dense(50, activation='LeakyReLU', name='discrim_layer2',kernel_constraint=const))
        model.add(Dense(1, activation='tanh', name='discrim_layer3'))
        # compile model
        optimizer = RMSprop(
            learning_rate=0.0001,
            rho=0.9,
            momentum=0,
            epsilon=1e-07,  # makes training more stable
            centered=True,  # makes training more stable
            clipvalue=0.01  # gradients are cliped to -0.01 - 0.01
        )
        # optimizer = Adam(
        #     learning_rate=0.001,
        #     beta_1=0.4,
        #     amsgrad=True
        # )
        model.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])
        self.discriminator = model

    def define_gan(self) -> None:
        """
        Create the GAN model. The GAN model architecture
        involves two sub-models: a generator model for
        generating new examples and a discriminator model
        for classifying whether generated examples are real,
        from the domain, or fake, generated by
        the generator model.
        """
        # make weights in the discriminator not trainable
        # generatir feeding in to disc
        self.discriminator.trainable = False
        model = Sequential(name='GAN')
        # add generator
        model.add(self.generator)
        # add the discriminator
        model.add(self.discriminator)
        # compile model
        optimizer = RMSprop(
            learning_rate=0.0001,
            rho=0.9,
            momentum=0,
            epsilon=1e-07,  # makes training more stable
            centered=True,  # makes training more stable
            clipvalue=0.01  # gradients are cliped to -0.01 - 0.01
        )
        model.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        self.gan = model

    def generate_real_samples(self, data: pd.DataFrame, num_rows: int) -> Tuple[pd.DataFrame, np.ndarray]:
        df2 = data.copy()
        X = df2.sample(num_rows)
        y = np.ones((num_rows, 1))
        return X, y

    def generate_simulated_samples(self, num_rows: int) -> Tuple[np.ndarray, np.ndarray]:
        x_input = self.generate_random_noise(num_rows)
        X = self.generator.predict(x_input)
        y = -np.ones((num_rows, 1))
        return X, y

    def preprocess(self, data: pd.DataFrame, keep_specials: bool = False) -> pd.DataFrame:
        """
        preprocessing the data before inputti ng, this includes
        :param data:
        :param keep_specials: keep special cases instead of dropping them (default False)
        :return:
        """
        data = data.copy()
        self.detect_categories(data)
        self.encode_category(data)
        data_without_specials = self.special_cases(data)
        if not keep_specials:
            data = data_without_specials
        data = self.encode_ordinals(data)

        # # future work for class weight
        # effort_sizes = data[self.ordinal_feature].value_counts()
        # boundary_counts = (effort_sizes.shift(-1) + effort_sizes).iloc[:-1]
        # self.class_weights = dict(enumerate(boundary_counts.sum() / boundary_counts / boundary_counts.count(), 1))
        # self.class_weights[0] = 1
        #
        # self.class_weights = None  # Disable for now

        self.scaler_income = QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')
        data[self.continuous_feature] = self.scaler_income.fit_transform(data[[self.continuous_feature]]).squeeze()

        return data

    def detect_categories(self, data):
        cats = data[self.ordinal_feature].astype('category')
        # self.effort_bands = cats.unique().sort_values()
        self.effort_bands = cats.cat.categories.sort_values()
        self.data_categories = cats.cat.categories

    def encode_ordinals(self, data):
        cat_cols = to_ordinal(data[self.ordinal_feature].cat.codes, num_classes=len(self.data_categories))
        encoded_data = pd.concat([data[[self.continuous_feature]], pd.DataFrame(cat_cols, index=data.index)], axis=1)
        return encoded_data

    def encode_category(self, data):
        data[self.ordinal_feature] = data[self.ordinal_feature].astype('category')
        data[self.ordinal_feature].cat.set_categories(self.effort_bands, ordered=True)

    def fit_to_data(self, data: pd.DataFrame, num_iterations: int) -> None:
        """
        Train generator and discriminator outputs
        :param data: input data
        :param num_iterations: number of iterations to run
        """
        # determine half the size of one batch, for updating the  discriminator
        d_history = []
        g_history = []
        # manually enumerate epochs
        for epoch in range(num_iterations):
            print(f'{epoch=}')
            for i in range(5):
                d_loss, d_loss_fake, d_loss_real = self.train_discriminator(data)
            g_loss_fake = self.train_generator()
            print('>%d, d1=%.3f, d2=%.3f d=%.3f g=%.3f' % (epoch + 1, d_loss_real, d_loss_fake, d_loss, g_loss_fake))

            d_history.append(d_loss)
            g_history.append(g_loss_fake)
            print(f'this is {d_loss=}')
            print(f'this is {g_loss_fake=}')

        self.d_history.extend(d_history)
        self.g_history.extend(g_history)
        self.plot_history()

    def train_generator(self):
        x_gan = self.generate_random_noise(self.batch_size)
        y_gan = np.ones((self.batch_size, 1))
        # update the generator via the discriminator's error
        self.discriminator.trainable = False
        g_loss_fake = self.gan.train_on_batch(x_gan, y_gan)
        return g_loss_fake

    def train_discriminator(self, data: pd.DataFrame):
        """
        Train the discriminator
        :param data: Original data
        """
        # prepare real samples
        x_real, y_real = self.generate_real_samples(data, self.batch_size)
        # prepare fake examples
        x_fake, y_fake = self.generate_simulated_samples(self.batch_size)
        # update discriminator
        self.discriminator.trainable = True
        d_loss_real, d_real_acc = self.discriminator.train_on_batch(x_real,
                                                                    y_real)  # , class_weight=self.class_weights)
        d_loss_fake, d_fake_acc = self.discriminator.train_on_batch(x_fake,
                                                                    y_fake)  # , class_weight=self.class_weights)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        return d_loss, d_loss_fake, d_loss_real

    def postprocess(self, X: np.ndarray) -> pd.DataFrame:
        """
        Postprocessing the generated data to a compatable shape to compare with original data
        :param X: generate data numpy array
        :return: dataframe fake data
        """

        data_fake = pd.DataFrame(data=X, columns=[self.continuous_feature] + [f'{n}' for n in range(self.cat_boundaries)])

        data_fake[self.continuous_feature] = self.scaler_income.inverse_transform(data_fake[[self.continuous_feature]]).squeeze()

        data_fake[self.ordinal_feature] = (data_fake.loc[:, '0':] > 0.5).sum(axis=1)
        data_fake[self.ordinal_feature] = data_fake[self.ordinal_feature].map(dict(enumerate(self.data_categories)))

        data_fake = data_fake[[self.continuous_feature, self.ordinal_feature]]

        sprinkle_rand = pd.DataFrame({'r': np.random.uniform(size=len(X))})
        sprinkle_overrides = pd.merge_asof(sprinkle_rand.sort_values('r'), self.specials_cdf, left_on='r', right_index=True,
                                           direction='forward').dropna()
        data_fake.loc[sprinkle_overrides.index] = sprinkle_overrides

        return data_fake

    def wasserstein_loss(self, y_true, y_pred):
        """
        Calculates the Wasserstein loss for a sample batch.
        :param y_true: actual distribution
        :param y_pred: simulated distribution
        :return: w_loss
        """
        # y_pred is the output from the discriminator, but it has a mixture of real and fake samples.
        # The real ones are labelled y_true==1; the fake ones are labelled y_true==-1
        # Considering just the real ones, w_loss = E(1 * y_pred_real)
        # Considering just the fake ones, w_loss = E(-1 * y_pred_fake) = -E(y_pred_fake)
        # Combining these: w_loss = E(y_pred_real) - E(y_pred_fake)
        # We can combine these simply by multiplying by y_true to get the same effect
        w_loss = keras.backend.mean(y_true * y_pred)
        return -w_loss

    def special_cases(self, original_data: pd.DataFrame) -> pd.DataFrame:
        """
        Treating special cases of zeros and ones in the dataset
        :param original_data: dataframe that comes in
        """
        specials_dist = original_data[original_data[self.continuous_feature] <= 1].value_counts() / len(original_data)
        self.specials_cdf = specials_dist.cumsum().reset_index().set_index('count', drop=True)
        # self.specials_cdf.index = pd.IntervalIndex.from_breaks([0] + list(self.specials_cdf.index))
        original_data = original_data[(original_data[self.continuous_feature] > 1)]
        return original_data

    def evaluate(self, original_data: pd.DataFrame, fake_data: pd.DataFrame, n_sample: int = 1000) -> \
            Tuple[np.float32, List[Tuple[float, float]]]:
        """
        Evaluate how close is the simulated data to the original data
        :param original_data: original data in dataframe form
        :param fake_data: generated data in dataframe form
        :param n_sample: numebr of samples to apply statistical test
        :return:
        """
        # table_evaluator = TableEvaluator(data, fake_data[[self.continuous_feature,self.ordinal_feature]], cat_cols=[self.ordinal_feature])
        # # table_evaluator.visual_evaluation()

        original_data = original_data.sample(n_sample)
        fake_data = fake_data.sample(n_sample)

        # Calculate Maximum Mean Discrepancy(MMD)
        self.encode_category(original_data)
        self.encode_category(fake_data)
        real_data = tf.convert_to_tensor(self.encode_ordinals(original_data), dtype=tf.float64)
        generated_data = tf.convert_to_tensor(self.encode_ordinals(fake_data), dtype=tf.float64)

        mmd_loss = self.compute_mmd(real_data, generated_data)

        # Calculating Kolmogorov-Smirnov (KS) test
        ks_test = self.calculate_ks_test(original_data, fake_data)

        print(f'MMD = {mmd_loss}')
        print(f'ks_test [Monthly Income, effort] = {ks_test}')

        # Ploting
        original_data['source'] = 'Real'
        fake_data['source'] = 'Fake'

        data_concat = pd.concat([original_data, fake_data]).sort_values(self.ordinal_feature)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        data_concat['counter'] = data_concat.groupby('source')[self.continuous_feature].cumcount()

        plot = sns.scatterplot(data=data_concat, hue='source', x=self.ordinal_feature, y='counter', edgecolor='none', ax=axes[0])
        plt.setp(plot.get_xticklabels(), rotation=45)

        sns.ecdfplot(data=data_concat, x=self.continuous_feature, hue='source', ax=axes[1])
        plt.show()

        return mmd_loss.numpy(), ks_test

    @staticmethod
    def compute_mmd(x: tf.Tensor, y: tf.Tensor, gamma: float = 0.001) -> tf.Tensor:
        # Pairwise squared Euclidean distances between samples
        xx = tf.matmul(x, x, transpose_b=True)
        yy = tf.matmul(y, y, transpose_b=True)
        xy = tf.matmul(x, y, transpose_b=True)

        # Gaussian kernel matrix
        kernel_xx = tf.exp(-gamma * tf.reduce_sum(xx, axis=-1))
        kernel_yy = tf.exp(-gamma * tf.reduce_sum(yy, axis=-1))
        kernel_xy = tf.exp(-gamma * tf.reduce_sum(xy, axis=-1))

        # MMD estimation
        mmd = tf.reduce_mean(kernel_xx) - 2 * tf.reduce_mean(kernel_xy) + tf.reduce_mean(kernel_yy)

        return mmd

    @staticmethod
    def calculate_ks_test(data: pd.DataFrame, fake_data: pd.DataFrame):
        """
        Calculate the Kolmogorov-Smirnov (KS) test statistics for each column in real and synthetic tabular datasets.
        :param fake_data: Simulated data
        :param data: Original data
        :return:
        """
        ks_statistics = []

        for column in data.columns:
            ks_statistic, p_value = ks_2samp(data[column], fake_data[column])
            ks_statistics.append((ks_statistic, p_value))

        return ks_statistics

    def plot_history(self):
        # plot loss
        plt.subplot(1, 1, 1)
        plt.plot(self.d_history, label='discriminator')
        plt.plot(self.g_history, label='generator')
        plt.legend()
        plt.show()
        plt.close()

    def save(self, name) -> None:
        self.generator.save(name)

    @staticmethod
    def load(name) -> None:
        load_model(name, compile=False).astype(int)

    def generate_random_noise(self, num_rows: int):
        """Generate noise
        :param num_rows: number of rows
        :return: noise sample
        """
        noise = randn(self.noise_dimensions * num_rows)
        noise = noise.reshape(num_rows, self.noise_dimensions)
        return noise


def main():
    data = pd.read_csv('original_data.csv')

    syn = Synthesizer.from_data_spec(dict(ordinal_feature='effort', continuous_feature='MonthlyIncome'))

    X_train, X_test, y_train, y_test = train_test_split(data, np.ones((len(data), 1)), test_size=0.2,
                                                        stratify=data[syn.ordinal_feature])

    tf.keras.utils.plot_model(syn.discriminator, to_file='discriminator.png')
    tf.keras.utils.plot_model(syn.generator, to_file='generator.png')
    tf.keras.utils.plot_model(syn.gan, to_file='gan.png')

    syn.learn(X_train, 4000)

    fake_data = syn.generate(10000)
    mmd_loss, ks_test = syn.evaluate(X_test, fake_data)
    print()
sns.displot(data =fake_data, x = 'MonthlyIncome', col = 'effort', col_wrap = 3, bins = 100)


if __name__ == '__main__':
    main()
