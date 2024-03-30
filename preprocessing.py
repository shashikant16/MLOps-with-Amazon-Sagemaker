
import pandas as pd
import numpy as np
import argparse
import os
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
def _parse_args():

    parser = argparse.ArgumentParser()
    
    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/')
    parser.add_argument('--filename', type=str, default='train_data.gzip')
    parser.add_argument('--outputpath', type=str, default='/opt/ml/processing/output/')

    return parser.parse_known_args()

# Defining a function to extract the top k features 
def get_top_k_features(X, Y, k):
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(X, Y)
        feature_df = pd.DataFrame(
            data=(X.columns, clf.feature_importances_)
        ).T.sort_values(by=1, ascending=False)
        cols = feature_df.head(k)[0].values
        return cols

if __name__=="__main__":
    # Process arguments
    args, _ = _parse_args()
    # Load data
    path = os.path.join(args.filepath,args.filename)
    print(path)
    
    # Reading the dataset and performing label encoding 
    df = pd.read_parquet(os.path.join(args.filepath,args.filename))
    le = LabelEncoder()
    df['Activity'] = le.fit_transform(df['Activity'])
    df.drop(['date_time'],axis =1 ,inplace = True)

    # Assignining the indepeneded and depended variable 
    X = df.drop(['Activity'], axis =1)
    Y = df['Activity']
    
    # Extracting top 12 important feature and filtering the dataset
    k =12
    final_cols = get_top_k_features(X, Y, k)
    final_cols = np.append(final_cols,np.array(['Activity']))
    df = df[final_cols]
    
    # Train, test, validation split
    # Randomly sort the data then split out first 70%, second 20%, and last 10%
    train_data, validation_data, test_data = np.split(df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.9 * len(df))])  
    
    # Storing of train, validation and test datasets 
    pd.concat([train_data['Activity'], train_data.drop(['Activity'], axis=1)], axis=1).to_csv(os.path.join(args.outputpath, 'train/train.csv'), index=False, header=False)
    pd.concat([validation_data['Activity'], validation_data.drop(['Activity'], axis=1)], axis=1).to_csv(os.path.join(args.outputpath, 'validation/validation.csv'), index=False, header=False)
    test_data[['Activity']].to_csv(os.path.join(args.outputpath, 'test/test_y.csv'), index=False, header=False)
    test_data.drop(['Activity'], axis=1).to_csv(os.path.join(args.outputpath, 'test/test_x.csv'), index=False, header=False)
    
    ## Save Features columns
    dump(final_cols, os.path.join(args.outputpath, 'feature/feature.joblib'))
    ## Save Encoder
    dump(le, os.path.join(args.outputpath, 'feature/encoder.joblib'))
    
    print("## Processing complete. Exiting.")
