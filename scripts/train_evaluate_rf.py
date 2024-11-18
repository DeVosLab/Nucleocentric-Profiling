from argparse import ArgumentParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve, PrecisionRecallDisplay
from matplotlib  import pyplot as plt
from sklearn.metrics import RocCurveDisplay
import math
import random
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns


def main(args):
    df = pd.read_csv(args.data_file)
    df_GT = pd.read_csv(args.GT_data_file)
    df_GT = df_GT[['ROI_img_name','true_condition']] 
    df_merged = pd.merge(df, df_GT, on='ROI_img_name')  #merge the true condition with the feature dataframe
    mixed = args.mixed
    if mixed == 'No':                       # depending on whether you want monocultures or cocultures, you need either the true condition or the target (layout)
        df = df_merged.loc[df_merged['target'] != 'co-culture'] 
    elif mixed =='Yes':
        df = df_merged.loc[df_merged['target'] == 'co-culture'] 
        df = df[df['true_condition'].str.contains('inconclusive')==False]
        df = df.drop(['target'], axis = 1)
        df.rename(columns = {'true_condition':'target'}, inplace = True)
    else:
        print("Error: do you want mixed or monoculture?")
    ord_enc = OrdinalEncoder()
    df["target"] = ord_enc.fit_transform(df[["target"]])  # 'target' is the variable which we want to predict


## training and testing stratified per region
    colnames = list(df.columns)# get column names to stratify per region
    region = args.regions2use
    print(region)
    if region == 'nucleus':
        nucleus = [col for col in colnames if "nucleus" in col]
        nucleus.append('true_density'); nucleus.append('target'); nucleus.append('ROI_img_name')
        df = df[nucleus]
    if region == 'cyto':
        cyto = [col for col in colnames if "cyto" in col]
        cyto.append('true_density'); cyto.append('target'); cyto.append('ROI_img_name')
        df = df[cyto]
    if region == 'cell':
        cell = [col for col in colnames if "cell" in col]
        cell.append('true_density'); cell.append('target'); cell.append('ROI_img_name')
        df = df[cell]
    if region == 'all':
        if df.columns.str.contains('ratio_astro').any():
            df = df.drop(['density','ratio_astro','ratio_SHSY5Y'], axis = 1)
        else:
            df = df


## training and testing stratified per channel
    colnames = list(df.columns)# get column names to stratify per region
    channel = args.channels2use
    print(channel)
    if channel == 'DAPI':
        DAPI = [col for col in colnames if "DAPI" in col]
        DAPI.append('true_density'); DAPI.append('target'); DAPI.append('ROI_img_name')
        df = df[DAPI]
    if channel == 'FITC':
        FITC = [col for col in colnames if "FITC" in col]
        FITC.append('true_density'); FITC.append('target'); FITC.append('ROI_img_name')
        df = df[FITC]
    if channel == 'Cy3':
        Cy3 = [col for col in colnames if "Cy3" in col]
        Cy3.append('true_density'); Cy3.append('target'); Cy3.append('ROI_img_name')
        df = df[Cy3]
    if channel == 'Cy5':
        Cy5 = [col for col in colnames if "Cy5" in col]
        Cy5.append('true_density'); Cy5.append('target'); Cy5.append('ROI_img_name')
        df = df[Cy5]
    if channel == 'all':
        if df.columns.str.contains('ratio_astro').any():
            df = df.drop(['density','ratio_astro','ratio_SHSY5Y'], axis = 1)
        else:
            df = df

    if args.drop_redundant == True:
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        df.drop(to_drop, axis=1, inplace=True)

    # sample and stratify train/validation/test set
    sampling = args.sample
    df = df.groupby('target', group_keys=False).apply(lambda x: x.sample(sampling))
    print(df)
    df['well'] = df['ROI_img_name'].str[:-8]
    wells = df['well'].unique()
    print(wells)
    n_test_wells = math.ceil(len(df['well'].unique())/3)
    test_wells = random.choices(wells, k = n_test_wells)
    test_df = df[df['well'].isin(test_wells)]
    test_df = test_df.dropna(axis=0)
    
    train_df = df[~df['well'].isin(test_wells)]
    train_df = train_df.drop(['ROI_img_name'], axis = 1) 

    train_df = (train_df.loc[ : , train_df.columns != 'target']
      .groupby('well').transform(lambda x: (x - x.mean()) / x.std())
      .join(train_df['target'])
    )
    train_df['true_density'] = train_df['true_density'].astype('float')
    train_df = train_df.dropna()
    print(train_df.columns)

    #stratification in validation, train and test set
    length = len(train_df.columns)-1
    attr = train_df.iloc[:, 0:length].values  #must be number of columns!
    labs = train_df.iloc[:, length].values  #must be number of columns!
    
    x_train, x_val, y_train, y_val = train_test_split(attr, labs, test_size=0.1)
    print(y_train)
    resampler = RandomUnderSampler(random_state = 42)
    x_train, y_train = resampler.fit_resample(x_train, y_train); print('Resampled train dataset shape %s' % Counter(y_train))
    x_val, y_val = resampler.fit_resample(x_val, y_val); print('Resampled val dataset shape %s' % Counter(y_val))

    # Find optimal number of trees
    accuracy_ntree = pd.DataFrame(columns=['ntree', 'accuracy'])  # create empty dataframe to store accuracy per number of trees
    x_tree, x_tree_test, y_tree, y_tree_test = train_test_split(x_val, y_val, test_size=0.3, random_state=0)  #within the validation set, you need stratification between training and test data for each model/nr of trees
    for n in range(10,151):  # loop over number of trees between 1 and 100
        clf = RandomForestClassifier(max_depth=10, random_state=0, n_estimators= n)  # define classifier
        clf.fit(x_tree, y_tree)    # train classifier
        y_pred_tree = clf.predict(x_tree_test)   # predict classifier on test data
        acc = accuracy_score(y_tree_test, y_pred_tree)   # determine accuracy by comparing prediction to known category
        add = [n, acc]  # append to the dataframe
        accuracy_ntree.loc[len(accuracy_ntree)] = add
    accuracy_ntree.plot.line(x = 'ntree', y = 'accuracy')  # plot the accuracy per number of trees to determine the optimal number of ntree
    plt.show()
    
    ntree = 30
    print(f'Optimal number of trees in the forest: {ntree}')   # enter in the terminal the number of trees you want to use for further analysis
    

    # Train RF model
    clf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators= ntree)
    clf.fit(x_train, y_train)
    
    # test model    
    print(test_df.columns)
    test_df = test_df.drop(['ROI_img_name'], axis = 1)
    test_df = test_df.dropna(axis=0)
    test_df = (test_df.loc[ : , test_df.columns != 'target']
      .groupby('well').transform(lambda x: (x - x.mean()) / x.std())
      .join(test_df['target'])
    )
    test_df = test_df.dropna()
    print(test_df)
    length = len(test_df.columns)-1

    x_test = test_df.iloc[:, 0:length].values  #must be number of columns!
    y_test = test_df.iloc[:, length].values  #must be number of columns!

    x_test, y_test = resampler.fit_resample(x_test, y_test); print('Resampled test dataset shape %s' % Counter(y_test))

    ## Get predictions
    y_pred = clf.predict(x_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

    # Precision-Recall curve
    y_pred_proba = clf.predict_proba(x_test)[::,1]
    prec,recall,_ = precision_recall_curve(y_true=y_test , probas_pred= y_pred_proba, pos_label=1)
    disp = PrecisionRecallDisplay(precision=prec, recall=recall)
    disp.plot()
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.title("Precision Recall curve")
    plt.plot([0, 1], [0, 0], color="navy", lw=2, linestyle="--")
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.show()
    
    # ROC curve
    RocCurveDisplay.from_predictions(y_test, y_pred_proba, pos_label=1)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.legend(loc=4)
    plt.title("Receiver operating characteristic (ROC) curve")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.show()
    
    # Confusion matrix
    CM = confusion_matrix(y_test, y_pred)
    CM = CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(CM, annot=True, annot_kws={'size':10}, cmap="Blues")
    class_names = ['1321N1', 'SHSY5Y']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model')
    plt.tight_layout()
    plt.show()
    
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True,
        help='file containing all features')
    parser.add_argument('--regions2use', type=str, required=True,
        help='Which region you want to analyse? Nucleus, cyto, cell or all')
    parser.add_argument('--channels2use', type=str, required=True,
        help='Which input channels do you want to use? DAPI, FITC, Cy3, Cy5 or all')
    parser.add_argument('--mixed', type=str, required=True,
        help='Classification of monoculture or coculture? Yes or No')
    parser.add_argument('--sample', type=int, required=True,
        help='number of instances per group used for training')
    parser.add_argument('--random_seed', type=int, required=True,
        help='train/validation/test random split')
    parser.add_argument('--GT_data_file', type=str, required=True,
        help='file containing ground truth')
    parser.add_argument('--drop_redundant', action='store_true', 
		help='Drop correlating features')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args)