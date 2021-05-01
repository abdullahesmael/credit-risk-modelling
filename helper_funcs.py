from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn import metrics
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier


def train_val_test_split(X, y, train=0.6, val=0.2, test=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test, random_state=1)
    val_ratio = val/(1-test)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=1)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, X_val, y_train, y_val, feature_cols=None):
    dtrain = lgb.Dataset(X_train, label=y_train, silent=True)
    dvalid = lgb.Dataset(X_val, label=y_val, silent=True)

    param = {'verbose': -1, 'num_leaves': 64, 'objective': 'binary',
             'metric': 'auc', 'seed': 7}
    num_round = 1000
    bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid],
                    early_stopping_rounds=20, verbose_eval=False)
    valid_pred = bst.predict(X_val)
    valid_score = metrics.roc_auc_score(y_val, valid_pred)
    # print(f"Validation AUC score: {valid_score}")
    return bst, valid_score


def encode(X_train,  X, encoding, y_train=None):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    cat_features = list(X_train.select_dtypes(exclude=numerics).columns.values)

    encoded = pd.DataFrame(index=X.index)
    if encoding == 'Label':
        encoder = LabelEncoder()
        for feature in cat_features:
            encoder.fit(X_train[feature])
            encoded[feature] = pd.DataFrame(
                data=encoder.transform(X[feature]), index=X.index)
            encoded[feature] = encoder.transform(X[feature])
    elif encoding == 'Count':
        encoder = ce.CountEncoder()
        encoder.fit(X_train[cat_features])
        encoded = encoder.transform(X[cat_features])
    elif encoding == 'Target':
        if y_train is None:
            raise ValueError('Request Target(y)')
        encoder = ce.TargetEncoder(cols=cat_features)
        encoder.fit(X_train[cat_features], y_train)
        encoded = encoder.transform(X[cat_features])
    elif encoding == 'CatBoost':
        if y_train is None:
            raise ValueError('Request Target(y)')
        encoder = ce.CatBoostEncoder(cols=cat_features)
        encoder.fit(X_train[cat_features], y_train)
        encoded = encoder.transform(X[cat_features])
    X = X[X.columns.difference(cat_features)]
    X = X.join(encoded)
    # encoded = pd.DataFrame(data=encoded, index=X.index.values)
    return X


def evaluate_encoder(X_train,  X_val, encoding, y_val, y_train=None):
    if (encoding == 'Label' or encoding == 'Count'):
        X_train_encoded = encode(X_train, X_train, encoding)
        X_val_encoded = encode(X_train, X_val, encoding)
    elif (encoding == 'Target' or encoding == 'CatBoost'):
        X_train_encoded = encode(X_train, X_train, encoding, y_train)
        X_val_encoded = encode(X_train, X_val, encoding, y_train)
    else:
        raise ValueError('Encoding must be in: Label,Count,Target,CatBoost')
    return train_model(X_train_encoded, X_val_encoded, y_train, y_val)[1]


def scale(X_train,  X, scaling):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_feats = list(X_train.select_dtypes(include=numerics).columns.values)
    if scaling == 'MinMax':
        scaler = MinMaxScaler()
    elif scaling == 'Standard':
        scaler = StandardScaler()
    scaler.fit(X_train[num_feats])
    scaled = pd.DataFrame(data=scaler.transform(X[num_feats]), index=X.index)
    X = X[X.columns.difference(num_feats)]
    X = X.join(scaled)
    return X


def evaluate_scaler(X_train,  X_val, scaling, y_val, y_train=None):
    X_train_scaled = scale(X_train, X_train, scaling)
    X_val_scaled = scale(X_train, X_val, scaling)
    return train_model(X_train_scaled, X_val_scaled, y_train, y_val)[1]


def handle_german_data_vars(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    cat_features = list(df.select_dtypes(exclude=numerics).columns.values)

    df_2 = pd.DataFrame(index=df.index)
    for column in cat_features:
        if column != 'personal_status':
            df_2[column] = eval('handle_german_'+column + "(df[column])")
        else:
            df_2['gender'] = eval('handle_german_'+column + "(df[column])")[0]
            df_2['marital_status'] = eval(
                'handle_german_'+column + "(df[column])")[1]
    X = df[df.columns.difference(cat_features)]
    X = X.join(df_2)
    return X


def handle_german_check_acc(column):
    mapping = {
        'A11': 'little',
        'A12': 'moderate',
        'A13': 'quite rich',
        'A14': 'rich',
    }
    return column.map(mapping)


def handle_german_credit_hist(column):
    mapping = {
        'A30': 'NO_CREDIT',
        'A31': 'ALL_PAID',
        'A32': 'EXISTING_PAID',
        'A33': 'DELAY_IN_PAST',
        'A34': 'CRITICAL_ACC',
    }
    return column.map(mapping)


def handle_german_loan_intent(column):
    mapping = {
        'A40': 'CAR_NEW',
        'A41': 'CAR_USED',
        'A42': 'FURNITURE/EQUIPMENT',
        'A43': 'RADIO/TELEVISION',
        'A44': 'DOMESTIC/APPLIANCE',
        'A45': 'REPAIRS',
        'A46': 'EDUCATION',
        'A47': 'VACATION',
        'A48': 'RETRAINING',
        'A49': 'BUSINESS',
        'A410': 'OTHERS',
    }
    return column.map(mapping)


def handle_german_savings_acc(column):
    mapping = {
        'A61': 'little',
        'A62': 'moderate',
        'A63': 'quite rich',
        'A64': 'rich',
        'A65': 'no saving acc'
    }
    return column.map(mapping)


def handle_german_tenure(column):
    mapping = {
        'A71': 'UNEMPLOYED',
        'A72': 'LESS THAN A YEAR',
        'A73': 'BETWEEN 1 AND 4 YEARS',
        'A74': 'BETWEEN 4 AND 7 YEARS',
        'A75': 'GREATER THAN 7 YEARS',
    }
    return column.map(mapping)


def handle_german_personal_status(column):
    mapping_gender = {
        'A91': 'MALE',
        'A92': 'FEMALE',
        'A93': 'MALE',
        'A94': 'MALE',
        'A95': 'FEMALE',
    }
    mapping_marital_status = {
        'A91': 'SINGLE_DIVORCED',
        'A92': 'MARRIED_DIVORCED',
        'A93': 'SINGLE',
        'A94': 'MARRIED_WIDOWED',
        'A95': 'SINGLE',
    }
    return column.map(mapping_gender), column.map(mapping_marital_status)


def handle_german_debtor_guarantor(column):
    mapping = {
        'A101': 'NONE',
        'A102': 'CO_APPLICANT',
        'A103': 'GUARANTOR',
    }
    return column.map(mapping)


def handle_german_assets(column):
    mapping = {
        'A121': 'REAL_ESTATE',
        'A122': 'SAVINGS_AGREEMENT',
        'A123': 'CAR',
        'A124': 'NO_ASSETS',
    }
    return column.map(mapping)


def handle_german_other_installments(column):
    mapping = {
        'A141': 'BANK',
        'A142': 'STORES',
        'A143': 'NONE',
    }
    return column.map(mapping)


def handle_german_housing(column):
    mapping = {
        'A151': 'RENT',
        'A152': 'OWN',
        'A153': 'FOR_FREE'
    }
    return column.map(mapping)


def handle_german_employment(column):
    mapping = {
        'A171': 'UNEMPLOYED_UNSKILLED',
        'A172': 'UNSKILLED',
        'A173': 'SKILLED_EMPLOYEE',
        'A174': 'SELF_EMPLOYED_MANAGEMENT',
    }
    return column.map(mapping)


def handle_german_telephone(column):
    mapping = {
        'A191': 'NO',
        'A192': 'YES'
    }
    return column.map(mapping)


def handle_german_foreign_work(column):
    mapping = {
        'A201': 'NO',
        'A202': 'YES'
    }
    return column.map(mapping)


def interactions_credit_risk_ds(X):
    interactions = pd.DataFrame(index=X.index)
    interactions['home_ownership_loan_intent'] = X['home_ownership'] + \
        '_'+X['loan_intent']
    interactions['home_ownership_grade'] = X['home_ownership'] + \
        '_'+X['grade']
    interactions['loan_intent_grade'] = X['loan_intent'] + \
        '_'+X['grade']
    X_int = X.join(interactions, lsuffix='')
    return X_int


def feature_scores(X_train, y_train, score_func):
    if score_func == 'mutual_information_score':
        # configure to select all features
        fs = SelectKBest(score_func=mutual_info_classif, k='all')
    elif score_func == 'f_value_score':
        # configure to select all features
        fs = SelectKBest(score_func=f_classif, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    # X_test_fs = fs.transform(X_test)
    mapping = {}
    count = 0
    for column in X_train.columns:
        mapping[column] = fs.scores_[count]
        count += 1

    sorted_values = sorted(mapping.values(), reverse=True)  # Sort the values
    sorted_dict = {}

    for i in sorted_values:
        for k in mapping.keys():
            if mapping[k] == i:
                sorted_dict[k] = mapping[k]
                break

    return sorted_dict


def L1_feature_rankings(X_train, y_train):
    feature_rank = []
    for C in np.arange(1, 0.01, -0.01):
        list_difference = L1_feature_drop(X_train, y_train, C)
        rank_difference = set(list_difference) - set(feature_rank)
        feature_rank.extend(list(rank_difference))

    for C in np.arange(0.01, 0, -0.0001):
        list_difference = L1_feature_drop(X_train, y_train, C)
        rank_difference = set(list_difference) - set(feature_rank)
        feature_rank.extend(list(rank_difference))
    rank_difference = set(X_train.columns) - set(feature_rank)
    feature_rank.extend(list(rank_difference))
    return feature_rank


def L1_feature_drop(X_train, y_train, C):
    logistic = LogisticRegression(
        C=C, penalty="l1", solver='liblinear', random_state=7).fit(X_train, y_train)
    model = SelectFromModel(logistic, prefit=True)

    X_new = model.transform(X_train)
    selected_features = pd.DataFrame(model.inverse_transform(X_new),
                                     index=X_train.index,
                                     columns=X_train.columns)

    # Dropped columns have values of all 0s, keep other columns
    selected_columns = selected_features.columns[
        selected_features.var() != 0
    ]
    selected_columns = selected_features.columns[selected_features.var(
    ) != 0]
    # list_difference = [
    # item for item in selected_columns if item not in X_train.columns]
    set_difference = set(X_train.columns) - set(selected_columns)
    list_difference = list(set_difference)
    return list_difference


def metric_evaluation(X_train, X_val, y_train, y_val):
    score = train_model(X_train, X_val, y_train, y_val)[1]
    clf = LGBMClassifier()
    clf.fit(X_train, y_train, verbose=False)
    weighted_avg = classification_report(
        y_val, clf.predict(X_val), output_dict=True)['weighted avg']
    weighted_avg['auc score'] = score
    return weighted_avg


def evaluate_interactions(X_train, X_train_int, X_val, X_val_int, y_train, y_val):
    interactions = {}
    interactions['without_interactions'] = metric_evaluation(
        X_train, X_val, y_train, y_val)
    interactions['with_interactions'] = metric_evaluation(
        X_train_int, X_val_int, y_train, y_val)
    return interactions


def evaluate_f_value_selection(X_train, X_val, y_train, y_val):
    f_scores = feature_scores(
        X_train, y_train, X_val, 'f_value_score')
    ranking = list(f_scores.keys())
    f_value_eval = {}
    for i in range(len(ranking)):
        feats = ranking[0:len(ranking)-i]
        f_value_eval[len(ranking)-i] = metric_evaluation(X_train[feats],
                                                         X_val[feats], y_train, y_val)
    return f_value_eval


def evaluate_mi_score_selection(X_train, X_val, y_train, y_val):
    mi_scores = feature_scores(
        X_train, y_train, X_val, 'mutual_information_score')
    ranking = list(mi_scores.keys())

    mi_score_eval = {}
    for i in range(len(ranking)):
        feats = ranking[0:len(ranking)-i]
        mi_score_eval[len(ranking)-i] = metric_evaluation(X_train[feats],
                                                          X_val[feats], y_train, y_val)
    return mi_score_eval


def evaluate_l1_reg_selection(X_train, X_val, y_train, y_val):
    ranking = L1_feature_rankings(X_train, y_train)
    l1_score_eval = {}
    for i in range(len(ranking)):
        feats = ranking[0:len(ranking)-i]
        l1_score_eval[len(ranking)-i] = metric_evaluation(X_train[feats],
                                                          X_val[feats], y_train, y_val)
    return l1_score_eval


def evaluate_lgbm_score_selection(X_train, X_val, y_train, y_val):
    clf = LGBMClassifier()
    clf.fit(X_train, y_train, eval_set=[
            (X_val, y_val)], verbose=False)
    feat_importance = {}
    count = 0
    feat_scores = list(clf.feature_importances_)
    for column in X_train.columns:
        feat_importance[column] = feat_scores[count]
        count += 1

    sorted_values = sorted(feat_importance.values(),
                           reverse=True)  # Sort the values
    sorted_dict = {}

    for i in sorted_values:
        for k in feat_importance.keys():
            if feat_importance[k] == i:
                sorted_dict[k] = feat_importance[k]
                break

    ranking = list(sorted_dict.keys())
    lgbm_score_eval = {}
    for i in range(len(ranking)):
        feats = ranking[0:len(ranking)-i]
        lgbm_score_eval[len(ranking)-i] = metric_evaluation(X_train[feats],
                                                            X_val[feats], y_train, y_val)
    return lgbm_score_eval
