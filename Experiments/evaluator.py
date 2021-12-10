
import numpy as np
import pandas as pd
from sklearn.linear_model import *
from tensorflow.keras.layers import *
from helpers import mean_squared_error, mean_absolute_error



def compute_GSS(y_pred_cls, y_true_cls):
    TP = np.sum(y_pred_cls & y_true_cls)
    FP = np.sum(y_pred_cls & ~y_true_cls)
    FN = np.sum(~y_pred_cls & y_true_cls)
    TP_rand = (TP + FN)*(TP + FP)/(len(y_true_cls))
    
    GSS = (TP - TP_rand)/(TP + FN + FP - TP_rand)
    return GSS

def get_best_GSS(iop_fold, thresh, sensor, config):
    
    Y_train_pred = iop_fold['Y_train_pred'][...,sensor]
    Y_val_pred = iop_fold['Y_val_pred'][...,sensor] 
    Y_train = iop_fold['Y_train'][...,sensor] 
    Y_val = iop_fold['Y_val'][...,sensor]
    
    if Y_train_pred.ndim > 1:
        f_start = config['eval_forecast_start']
        f_end = config['eval_forecast_end']
        
        Y_train_pred = Y_train_pred[:,f_start:f_end].ravel()
        Y_val_pred = Y_val_pred[:,f_start:f_end].ravel()
        Y_train = Y_train[:,f_start:f_end].ravel()
        Y_val = Y_val[:,f_start:f_end].ravel()
    
    Y_val_cls = Y_val > thresh
    Y_train_cls = Y_train > thresh
    
    GSS_list_train = []
    GSS_list_val = []
    # Round & unique til að minnka tíma fyrir stór n 
    sorted_pred = sorted(np.unique(np.round(Y_train_pred,1)))
    for t in sorted_pred:
        Y_train_pred_cls = Y_train_pred > t
        Y_val_pred_cls = Y_val_pred > t
        GSS_train = compute_GSS(Y_train_pred_cls,Y_train_cls)
        GSS_val = compute_GSS(Y_val_pred_cls,Y_val_cls)
        GSS_list_train.append(GSS_train)
        GSS_list_val.append(GSS_val)
        
    best_thresh = np.argmax(GSS_list_train)
    GSS_train_best = GSS_list_train[best_thresh]
    GSS_val_best = GSS_list_val[best_thresh]
    return GSS_train_best,GSS_val_best

# def plt_GSS(iop_fold, thresh,sensor):
#     Y_train_pred = iop_fold['Y_train_pred'][:,sensor]
#     Y_val_pred = iop_fold['Y_val_pred'][:,sensor] 
#     Y_train = iop_fold['Y_train'][:,sensor] 
#     Y_val = iop_fold['Y_val'][:,sensor]
    
#     Y_val_cls = Y_val > thresh
#     Y_train_cls = Y_train > thresh
    
#     GSS_list_train = []
#     GSS_list_val = []
#     sorted_pred = sorted(Y_train_pred)
#     for t in sorted_pred:
#         Y_train_pred_cls = Y_train_pred > t
#         Y_val_pred_cls = Y_val_pred > t
#         GSS_train = compute_GSS(Y_train_pred_cls,Y_train_cls)
#         GSS_val = compute_GSS(Y_val_pred_cls,Y_val_cls)
#         GSS_list_train.append(GSS_train)
#         GSS_list_val.append(GSS_val)
        
#     plt.plot(sorted_pred,GSS_list_train)
#     plt.plot(sorted_pred,GSS_list_val)
#     plt.legend(['Train','Validation'])
#     plt.show()

def get_average_best_GSS(iop,thresh,sensor, config):
    
    best_GSS_train_list = []
    best_GSS_val_list = []
    for fold in iop:

        iop_fold = iop[fold]

        GSS_train_best,GSS_val_best = get_best_GSS(iop_fold,thresh,sensor, config)
        best_GSS_train_list.append(GSS_train_best)
        best_GSS_val_list.append(GSS_val_best)
    average_GSS_train = np.mean(best_GSS_train_list)
    average_GSS_val = np.mean(best_GSS_val_list)
    return average_GSS_train,average_GSS_val

def CSI(results):
    return results['TP']/(results['TP'] + results['FP'] + results['FN'])


def CSI_per_sensor_per_threshold_aggregate(results):
    results = results.copy().drop(['fold', 'split'], axis=1)
    results_fine = results.groupby(['sensor', 'threshold', 'f_dist']).sum()
    results = results.groupby(['sensor', 'threshold', ]).sum()
    for index, row in results.iterrows():
        results.at[index, 'CSI'] = CSI(row)
    for index, row in results_fine.iterrows():
        results_fine.at[index, 'CSI'] = CSI(row)
    results = results.reset_index()
    results_fine = results_fine.reset_index()
    return results, results_fine


def CSI_per_sensor_per_threshold(results):
    train = results[results.loc[:, 'split'] == 'train']
    val = results[results.loc[:, 'split'] == 'val']
    results_train, results_train_fine = CSI_per_sensor_per_threshold_aggregate(
        train)
    results_val, results_val_fine = CSI_per_sensor_per_threshold_aggregate(val)
    return results_train, results_val, results_train_fine, results_val_fine


def CLF_error(Y, Y_pred, thresh):
    Y_bool = Y == 1
    Y_pred_bool = Y_pred > thresh
    TP = np.sum(Y_bool & Y_pred_bool, axis=0)  # True True
    FP = np.sum(~Y_bool & Y_pred_bool, axis=0)  # False True
    TN = np.sum(~Y_bool & ~Y_pred_bool, axis=0)  # False False
    FN = np.sum(Y_bool & ~Y_pred_bool, axis=0)  # True False
    return TP, FP, TN, FN


def compute_CLS(fold_data, all_thresholds):
    columns = ['fold', 'split', 'sensor',
               'threshold', 'f_dist', 'TP', 'FP', 'TN', 'FN']
    values = []
    fold = fold_data['fold']
    # For either sensor
    for sensor_id, sensor_thresholds in enumerate(all_thresholds):
        # for each threshold
        for threshold_id, threshold in enumerate(sensor_thresholds):
            # for compatability
            if threshold != 0.5:
                threshold_id = None

            Y_val_pred_clf = fold_data['Y_val_pred'][:,
                                                     :, sensor_id, threshold_id]
            Y_train_pred_clf = fold_data['Y_train_pred'][:,
                                                         :, sensor_id, threshold_id]
            Y_val_clf = fold_data['Y_val'][:, :, sensor_id, threshold_id]
            Y_train_clf = fold_data['Y_train'][:, :, sensor_id, threshold_id]

            # validation
            TP, FP, TN, FN = CLF_error(Y_val_clf, Y_val_pred_clf, threshold)
            for f_dist, (tp, fp, tn, fn) in enumerate(zip(TP, FP, TN, FN)):
                values.append(
                    [fold, 'val', sensor_id, threshold, f_dist, tp, fp, tn, fn])

            # training
            TP, FP, TN, FN = CLF_error(
                Y_train_clf, Y_train_pred_clf, threshold)
            for f_dist, (tp, fp, tn, fn) in enumerate(zip(TP, FP, TN, FN)):
                values.append([fold, 'train', sensor_id,
                              threshold, f_dist, tp, fp, tn, fn])

    # values = np.array(values)
    # print(values.shape,len(columns))
    df_all_results = pd.DataFrame(values, columns=columns)
    return df_all_results


def evaluate_loss(fold_data, evaluation_dict, loss_fn, weight_fn = None):
    loss_name = loss_fn.name
    fold = evaluation_dict['fold']
    loss_t = loss_fn(fold_data['Y_train'], fold_data['Y_train_pred'],weight_fn)
    loss_v = loss_fn(fold_data['Y_val'], fold_data['Y_val_pred'],weight_fn)

    weight_fn_str = ''
    if type(weight_fn) != type(None):
        weight_fn_str = 'weighted_'
    evaluation_dict[f'{weight_fn_str}{loss_name}'] =\
        evaluation_dict[f'{weight_fn_str}{loss_name}_fold_{fold}'] =\
        loss_t
    evaluation_dict[f'{weight_fn_str}val_{loss_name}'] = \
        evaluation_dict[f'{weight_fn_str}val_{loss_name}_fold_{fold}'] =\
        loss_v


def evaluate_MSE(fold_data, evaluation_dict):
    mse = mean_squared_error()
    evaluate_loss(fold_data, evaluation_dict, mse)


def evaluate_MAE(fold_data, evaluation_dict):
    mae = mean_absolute_error()
    evaluate_loss(fold_data, evaluation_dict, mae)

def evaluate_weighted_MSE(fold_data, evaluation_dict,weight_func):
    mse = mean_squared_error()
    evaluate_loss(fold_data, evaluation_dict, mse, weight_func)

def evaluate_weighted_MAE(fold_data, evaluation_dict,weight_func):
    mae = mean_absolute_error()
    evaluate_loss(fold_data, evaluation_dict, mae, weight_func)


def error_profile(fold_data, evaluation_dict, agg_func, name):
    fold = fold_data['fold']
    resids_train = fold_data['Y_train'] - fold_data['Y_train_pred']
    resids_val = fold_data['Y_val'] - fold_data['Y_val_pred']

    train_profile = np.mean(agg_func(resids_train), axis=(0, -1))
    val_profile = np.mean(agg_func(resids_val), axis=(0, -1))
    evaluation_dict[f'{name}_profile'] = evaluation_dict[f'{name}_profile_fold_{fold}'] = train_profile
    evaluation_dict[f'val_{name}_profile'] = evaluation_dict[f'val_{name}_profile_fold_{fold}'] = val_profile


def evaluate_MSE_profile(fold_data, evaluation_dict):
    error_profile(fold_data, evaluation_dict,
                  (lambda x: x**2), 'mean_squared_error')


def evaluate_MAE_profile(fold_data, evaluation_dict):
    error_profile(fold_data, evaluation_dict, np.abs, 'mean_absolute_error')


def evaluate_total_statistic(iop, results, statistic_fn, weight_func = None):
    y_preds = []
    y_trues = []
    for fold in iop:
        y_preds.append(iop[fold]['Y_val_pred'])
        y_trues.append(iop[fold]['Y_val'])
        
    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)

    statistic_fn_instance = statistic_fn(None)
    stat = statistic_fn_instance(y_preds, y_trues, weight_func)
    name = statistic_fn_instance.name
    if type(weight_func) != type(None):
        name = 'weighted_' + name
    
    stat_name = f'total_{name}'
    results[stat_name] = stat


def evaluate_total_MSE(iop, results):
    evaluate_total_statistic(iop, results, mean_squared_error)

def evaluate_total_MAE(iop, results):
    evaluate_total_statistic(iop, results, mean_absolute_error)

def evaluate_total_weighted_MSE(iop, results,weight_fn):
    evaluate_total_statistic(iop, results, mean_squared_error,weight_fn)

def evaluate_total_weighted_MAE(iop, results,weight_fn):
    evaluate_total_statistic(iop, results, mean_absolute_error,weight_fn)

def only_MSE(iop, weight_fn):
    results = {'folds': {},
               'summary': {}}
    
    # Summary statistics for whole dataset
    evaluate_total_MSE(iop, results['summary'])
    evaluate_total_MAE(iop, results['summary'])
    
    # Summary statistics for whole dataset
    evaluate_total_weighted_MSE(iop, results['summary'], weight_fn)
    evaluate_total_weighted_MAE(iop, results['summary'], weight_fn)

    return results



def only_MSE_and_GSS(iop,weight_fn,evaluation_config):
    results = {'folds': {},
               'summary': {}}
    
    # Summary statistics for whole dataset
    evaluate_total_MSE(iop, results['summary'])
    evaluate_total_MAE(iop, results['summary'])
    
    # Summary statistics for whole dataset
    evaluate_total_weighted_MSE(iop, results['summary'], weight_fn)
    evaluate_total_weighted_MAE(iop, results['summary'], weight_fn)


    _,average_GSS_val_0 = get_average_best_GSS(iop,500,0,evaluation_config)
    results['summary']['GSS_val_0'] = average_GSS_val_0
    
    _,average_GSS_val_1 = get_average_best_GSS(iop,90,1,evaluation_config)
    results['summary']['GSS_val_1'] = average_GSS_val_1
    results['summary']['GSS_val_average'] = (average_GSS_val_0 + average_GSS_val_1)/2


    return results

def evaluate_results(iop, thresholds, weight_fn):

    results = {'folds': {},
               'summary': {}}
    for fold in iop:
        fold_data = iop[fold]

        # Evaluation functions
        fold_evaluation_stats = {}
        fold_evaluation_stats['fold'] = fold

        # Over whole fold
        evaluate_MSE(fold_data, fold_evaluation_stats)
        evaluate_MAE(fold_data, fold_evaluation_stats)
        
        evaluate_weighted_MSE(fold_data, fold_evaluation_stats, weight_fn)
        evaluate_weighted_MAE(fold_data, fold_evaluation_stats, weight_fn)

        # For each prediction distance
        evaluate_MAE_profile(fold_data, fold_evaluation_stats)
        evaluate_MSE_profile(fold_data, fold_evaluation_stats)

        results['folds'][fold] = fold_evaluation_stats
    # Summary statistics for whole dataset
    evaluate_total_MSE(iop, results['summary'])
    evaluate_total_MAE(iop, results['summary'])
    
    # Summary statistics for whole dataset
    evaluate_total_weighted_MSE(iop, results['summary'], weight_fn)
    evaluate_total_weighted_MAE(iop, results['summary'], weight_fn)

    return results
