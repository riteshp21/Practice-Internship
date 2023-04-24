import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
from utils import save_object, label_encodings
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("../../artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        try:

            numerical_columns = ['Unit_of_Measure_Per_Pack', 'Line_Item_Quantity', 'Line_Item_Value',
                                 'Pack_Price', 'Unit_Price', 'Weight_Kilograms',
                                 'Line_Item_Insurance_USD']
            categorical_columns = ['Country', 'Managed_By', 'Fulfill_Via', 'Shipment_Mode',
                                   'Product_Group', 'Sub_Classification',
                                   'Brand', 'First_Line_Designation']# 'Item_Description'  , 'Vendor','Dosage', 'Dosage_Form',
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse_output=False,handle_unknown = 'ignore')),
                    ("scaler", StandardScaler(with_mean=False))])

            preprocessor = ColumnTransformer(
                [("num_pipeline", num_pipeline, numerical_columns),
                 ("cat_pipelines", cat_pipeline, categorical_columns)])
            # logging.info(f"{preprocessor}")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = 'Freight_Cost_USD'
            numerical_columns = ['Unit_of_Measure_Per_Pack', 'Line_Item_Quantity', 'Line_Item_Value',
                                 'Pack_Price', 'Unit_Price', 'Weight_Kilograms', 'Freight_Cost_USD',
                                 'Line_Item_Insurance_USD']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            #input_feature_train_df = label_encodings(input_feature_train_df)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            #label_encodings(input_feature_test_df)
            target_feature_test_df = test_df[target_column_name]
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            #preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
