def evaluate_model(model_constructor,
                   input_data,
                   output_data,
                   fn_crossval = KFold,
                   folds = 5,
                   tensorflow_model = False,
                   **fit_kwargs):
    
    model_list = []
    history = {}
    kfold = fn_crossval(folds)
    
    for i,(train, test) in enumerate(kfold.split(output_data)):
        X_train = [inp[train] for inp in input_data]
        X_test = [inp[test] for inp in input_data]
        Y_train = output_data[train]
        Y_test = output_data[test]

        fold_information = {}
        model = model_constructor()
        
        if tensorflow_model:
            model.fit(X_train,
                      Y_train,
                      validation_data = (X_test,Y_test),
                      **fit_kwargs)
        else:
            model.fit(X_train,
                      Y_train,
                      **fit_kwargs)
                    
        Y_train_pred = model.predict(X_train).ravel()
        Y_test_pred = model.predict(X_test).ravel()
        
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()
        
        fold_information['mse_train'] = tf.losses.MSE(Y_train,Y_train_pred).numpy()
        fold_information['mse_test'] = tf.losses.MSE(Y_test,Y_test_pred).numpy()
        fold_information['mae_train'] = tf.keras.metrics.MAE(Y_train,Y_train_pred).numpy()
        fold_information['mae_test'] = tf.keras.metrics.MAE(Y_test,Y_test_pred).numpy()
        
        fold_information['X_train'] = X_train
        fold_information['X_test'] = X_test
        fold_information['Y_train'] = Y_train
        fold_information['Y_test'] = Y_test
        fold_information['Y_train_pred'] = Y_train_pred
        fold_information['Y_test_pred'] = Y_test_pred
        
        
        model_list.append(model)
        
        history[i] = fold_information
    return history,model_list

