import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import joblib
import logging

np.random.seed(42)
tf.random.set_seed(42)

class airbnb_price_prediction:
    def __init__(self):
        """Initialize the model with necessary components."""
        self.scaler = StandardScaler()
        self.label_encoder = {}
        self.model = None
        self.feature_columns = None
        self.required_columns = ['price', 'rating', 'reviews', 'bathrooms', 'beds',
                               'guests', 'bedrooms', 'studios', 'toiles', 'country', 'host_name']

        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def load_data(self, file_path):
        """Load and validate the dataset from a CSV file.

        Args:
            file_path (str): Path to the CSV file containing Airbnb data.

        Returns:
            pd.DataFrame: The loaded dataset.

        Raises:
            FileNotFoundError: If the specified file doesn't exist.
            ValueError: If required columns are missing from the dataset.
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        self.df = pd.read_csv(file_path)

        # Validate required columns
        missing_cols = [col for col in self.required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.logger.info("Data loaded successfully")
        self.logger.info(f"Dataset shape: {self.df.shape}")
        self.logger.info(f"Columns: {list(self.df.columns)}")

        # Display basic information about the dataset
        print("\nDataset Information:")
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nColumn names: {list(self.df.columns)}")
        print(f"\nData types:\n{self.df.dtypes}")
        print("\nFirst few rows:")
        print(self.df.head())
        print("\nMissing values:")
        print(self.df.isnull().sum())
        print("\nSummary statistics:")
        print(self.df.describe())

        return self.df

    def eda(self):
        """Performing EDA"""

        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))

        plt.subplot(3,4,1)
        plt.hist(self.df['price'], bins=50, alpha=0.7, color='red')
        plt.title("Price distribution")
        plt.xlabel('Price')
        plt.ylabel('Frequency')

        plt.subplot(3,4,2)
        plt.hist(np.log1p(self.df['price']), bins=50, alpha=0.7, color='green')
        plt.title('Price distribution using log-transformed')
        plt.xlabel('Log(Price+1)')
        plt.ylabel('Frequency')

        plt.subplot(3,4,3)
        plt.boxplot(self.df['price'])
        plt.title("Price again with boxplot")
        plt.ylabel('Price')

        plt.subplot(3,4,4)
        plt.hist(self.df['rating'].dropna(), bins=50, alpha=0.7, color='yellow')
        plt.title("Rating")
        plt.xlabel('Rating')
        plt.ylabel('Frequency')

        plt.subplot(3,4,5)
        plt.hist(self.df['reviews'], bins=50, alpha=0.7, color='purple')
        plt.title('Review distribution')
        plt.xlabel('Reviews')
        plt.ylabel('Frequency')

        plt.subplot(3,4,6)
        bedroom_counts = self.df['bedrooms'].value_counts().sort_index()
        plt.bar(bedroom_counts.index, bedroom_counts.values, color='blue')
        plt.title('Bedrooms distribution')
        plt.xlabel('Bedroom counts')
        plt.ylabel('Count')

        plt.subplot(3,4,7)
        top_countries = self.df['country'].value_counts().head(10)
        country_prices = []
        for country in top_countries.index:
            country_prices.append(self.df[self.df['country'] == country]['price'].median())
        plt.bar(top_countries.index, country_prices, color='pink')
        plt.title('Median Price by Country (Top 10)')
        plt.xticks(rotation=45)
        plt.xlabel('Country')
        plt.ylabel('Median Price')

        plt.subplot(3,4,8)
        plt.scatter(self.df['guests'], self.df['price'], alpha=0.5, color='skyblue')
        plt.title('Price vs guest')
        plt.xlabel('Guest number')
        plt.ylabel('Prices')

        plt.subplot(3,4,9)
        price_of_bathroom = self.df.groupby('bathrooms')['price'].median()
        plt.bar(price_of_bathroom.index, price_of_bathroom.values, color='gold')
        plt.title('Median price of bathrooms')
        plt.xlabel('Bathrooms')
        plt.ylabel('Price in median')

        plt.subplot(3,4,10)
        plt.scatter(self.df['rating'], self.df['price'], alpha=0.5, color='orange')
        plt.title('Price vs rating')
        plt.xlabel('Rating')
        plt.ylabel('price')

        plt.subplot(3,4,11)
        plt.scatter(self.df['reviews'], self.df['price'], alpha=0.5, color='brown')
        plt.title('Price vs reviews')
        plt.xlabel('Reviews')
        plt.ylabel('Price')

        plt.subplot(3,4,12)
        cols = ['price', 'rating', 'reviews', 'bathrooms', 'beds', 'guests', 'bedrooms','studios']
        # Ensure all columns are numeric for correlation
        for col in cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        cor_matrix = self.df[cols].corr()
        sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', center=0, fmt='2f')
        plt.title('Correlation Matrix')

        plt.tight_layout()
        plt.show()

        ## corr insights
        print("\n")
        print("Correlation analysis")
        corr_price = cor_matrix['price'].sort_values(ascending=False)
        print("Price correlations:")
        print(corr_price)

    def cleaning_data(self):
        """Cleaning the data"""
        print("\n\n")
        self.df_processed = self.df.copy()

        """Handling missing data"""
        print("\n")


        ## we need to check for any missing values
        ## bathrooms, beds, guests, bedrooms, studios, toilets, ratings = replace it with median value
        if self.df_processed['rating'].isnull().any():
            median_rating = self.df_processed['rating'].median() ## filling missing data with median data
            self.df_processed['rating'].fillna(median_rating, inplace=True)
            print(f"{median_rating}")

        number_cols = ['bathrooms', 'beds', 'guests', 'bedrooms', 'studios', 'toiles']
        for cols in number_cols:
            if cols in self.df_processed.columns and self.df_processed[cols].isnull().any():
                median_value = self.df_processed[cols].median()
                self.df_processed[cols].fillna(median_value, inplace=True)
                print(f"{median_value}")

        ##
        categorical_col = ['country', 'host_name']
        for cols in categorical_col:
            if cols in self.df_processed.columns and self.df_processed[cols].isnull().any():
                mode_value = self.df_processed[cols].mode()[0]
                self.df_processed[cols].fillna(mode_value, inplace=True)
                print(f"filling missing {cols} with mode: {mode_value}")

        ## we need to remove the outliers from price
        print("\n")
        q1 = self.df_processed['price'].quantile(0.25)
        q3 = self.df_processed['price'].quantile(0.75)

        ## we can find the spread in the middle of the dataset
        ## interquartile range = q3 - q1
        ## lower bound = q1 - 1.5 * interquartile range
        ## upper bound opposite
        interquartile_range = q3 - q1
        lower_bound = q1 - 1.5 * interquartile_range
        upper_bound = q3 + 1.5 * interquartile_range

        initial_count = len(self.df_processed)
        self.df_processed = self.df_processed[
            (self.df_processed['price'] >= lower_bound) &
            (self.df_processed['price'] <= upper_bound)
        ]

        final_count = len(self.df_processed)
        print(f"{initial_count - final_count} outliers from price")


        ## in feature engineering, we need to get price of each guest, total rooms and review
        ## maybe price of each guest = price / guest
        ## total rooms = bedrooms + bathrooms
        ## reviews per rating = review - rating + 1 (0-5)
        print("\nFeature engineering")
        self.df_processed['guests'].replace(0, np.nan, inplace=True)
        self.df_processed['price_per_guest'] = self.df_processed['price'] / self.df_processed['guests']
        self.df_processed['total_rooms'] = self.df_processed['bedrooms'] + self.df_processed['bathrooms']
        self.df_processed['review_per_rating'] = self.df_processed['reviews'] / (self.df_processed['rating'] + 1)

        ## now we can separate the top countries from other countries
        ## maybe grouping other countries as others, we can create one
        ## for x in self.df_preprocessing['country]:
        ##     if x in top_countries:
        ##          self.df_preprocessing['encoded_country'] = x
        ##     else:
        ##          self.df_preprocessing['encoded_country'] = "Other"
        top_countries = self.df_processed['country'].value_counts().head(20).index
        self.df_processed['encoded_country'] = [
            x if x in top_countries else 'Other' for x in self.df_processed['country']
        ]

        ## performing one-hot encoding
        country_dump = pd.get_dummies(self.df_processed['encoded_country'], prefix='country')
        self.df_processed = pd.concat([self.df_processed, country_dump], axis=1)

        ## after that now we can select features for model training
        feature_columns = ['rating', 'reviews', 'bathrooms', 'beds', 'guests', 'bedrooms', 'studios', 'toiles', 'price_per_guest', 'total_rooms', 'review_per_rating']

        country_columns = [col for col in self.df_processed.columns if col.startswith('country_')]
        feature_columns.extend(country_columns)

        self.feature_columns = feature_columns
        print(f"{len(feature_columns)} features selected for model training: {feature_columns}")

        return self.df_processed

    def data_prep_modeling(self):
        print("\nPreparing data for modeling:")
        X = self.df_processed[self.feature_columns]
        y = self.df_processed['price']

        # Handle any remaining missing values
        X = X.fillna(X.median())

        # Split and scale the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        self.logger.info(f"Training set shape: {self.X_train_scaled.shape}")
        self.logger.info(f"Testing set shape: {self.X_test_scaled.shape}")

        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

    def ann_model(self, input_dim):
        """Creating the ANN model"""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_dim=input_dim),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(1, activation='linear')
        ])

        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=0.001),
            loss = 'mse',
            metrics = ['mae']
        )

        print("\nModel architecture: ")
        model.summary()

        return model

    def model_train(self):
        """Training the model now"""
        print("\nModel training: ")

        self.model = self.ann_model(self.X_train_scaled.shape[1])

        ## defining callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor = 'val_loss', patience = 10, restore_best_weights = True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_loss', factor = 0.5, patience = 5, min_lr = 1e-6
        )

        history = self.model.fit(
            self.X_train_scaled, self.y_train,
            epochs = 100,
            validation_split = 0.2,
            callbacks = [early_stopping, reduce_lr],
            verbose = 1
        )

        return history

    def model_evaluation(self, history):
        """in this process, we can try to evaluate the model and improve accuracy"""
        print("\n")

        ## i think at first we can flatten the training and test data
        ## afterwards we can calculate the metrics
        ## making prediction
        y_pred_train = self.model.predict(self.X_train_scaled).flatten()
        y_pred_test = self.model.predict(self.X_test_scaled).flatten()

        ## calculating the metrics
        train_mse = mean_squared_error(self.y_train, y_pred_train)
        test_mse = mean_squared_error(self.y_test, y_pred_test)

        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)

        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)

        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)

        ## plotting the training history
        plt.figure(figsize=(15,5))

        plt.subplot(1,3,1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()


        plt.subplot(1,3,2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        ## plotting the predicted vs actual value
        plt.subplot(1,3,3)
        plt.scatter(self.y_test, y_pred_test, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.main(),
                                                          self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted prices')
        plt.tight_layout()
        plt.show()

        return{
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_r2': train_r2, 'test_r2': test_r2
        }


    def model_ensemble(self):
        """Ensemble model for better accuracy"""
        print("\n")

        ## now we can just preform random forest regression, gradient boosting regression, and lastly ann prediction
        ## random forest regression
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        rf_pred = rf_model.predict(self.X_test)
        rf_rmse = np.sqrt(mean_squared_error(self.y_test, rf_pred))
        rf_r2 = r2_score(self.y_test, rf_pred)

        ## gradient boosting
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(self.X_train, self.y_train)
        gb_pred = gb_model.predict(self.X_test)
        gb_rmse = np.sqrt(mean_squared_error(self.y_test, gb_pred))
        gb_r2 = r2_score(self.y_test, gb_pred)

        ## ann prediction
        ann_pred = self.model.predict(self.X_test_scaled).flatten()

        ensemble_pred = (ann_pred + rf_pred + gb_pred) / 3
        ensemble_rmse = np.sqrt(mean_squared_error(self.y_test, ensemble_pred))
        ensemble_r2 = r2_score(self.y_test, ensemble_pred)

        print("\n")
        print("Model comparison: ")
        print(f"Random forest - rmse :{rf_rmse: .2f}, r squared: {rf_r2: .2f}")
        print(f"Gradient boosting - rmse: {gb_rmse: .2f}, r squared: {gb_r2: .2f}")
        print(f"Ensemble - rmse : {ensemble_rmse: .2f}, r squared: {ensemble_r2: .2f}")

        return ensemble_pred, ensemble_rmse, ensemble_r2

    def cross_val(self):
        print("\n")
        print("Cross validation score:")

        ## for cross validation, we can now have to make sure of keras as it needs special handling
        ## at first trying random forrest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        X_full = np.vstack([self.X_train, self.X_test])
        y_full = np.hstack([self.y_train, self.y_test])

        ## now we can 5 fold this
        cross_val_scores = cross_val_score(rf_model, X_full, y_full, cv=5)
        cross_val_rmse_score = np.sqrt(-cross_val_scores)

        print(f"Cross validation rmse score: {cross_val_rmse_score}")
        print(f"mean cross validation rmse: {cross_val_rmse_score.mean():.2f} (+/-{cross_val_rmse_score.std() * 2: .2f})")

        return cross_val_rmse_score

def main():
    """main function"""
    print("\nPrice prediction of Airbnb: ")
    print("\n")
    model = airbnb_price_prediction()

    data_path = 'airbnb.csv'
    model.load_data(data_path)
    model.eda()
    model.cleaning_data()
    model.data_prep_modeling()
    model.model_train()
    model.evaluate_model()

if __name__ == "__main__":
    main()
