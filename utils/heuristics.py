from tqdm.auto import tqdm
import atpbar

def heuristic_search_per_class(dataframe, param_range, results_arr, results_arr_ps, results_arr_rev, results_arr_ps_rev, label):
    # trues, preds_a, preds_b, preds_score_a, preds_score_b, param_range,
    acc = []
    acc_ps = []
    acc_rev = []
    acc_ps_rev = []

    # df_size = dataframe.shape[0]
    df_label_a = dataframe[dataframe['classification_a'] == label]
    df_label_b = dataframe[dataframe['classification_b'] == label]
    # print(len(df))

    # for thr in tqdm(param_range):
    for thr in param_range:
        max_p_correct = 0
        max_p_correct_ps = 0
        max_p_model_b_usage = 0


        max_p_correct_rev = 0
        max_p_correct_ps_rev = 0
        max_p_model_b_usage_rev = 0
        
        for true, pred_a, pred_b, pred_score_a, pred_score_b in zip(df_label_a['true'], df_label_a['classification_a'], df_label_a['classification_b'], df_label_a['max_prob_a'], df_label_a['max_prob_b']):
            
            pred = pred_a
            pred_ps = pred_a

            if pred_score_a < thr:
                max_p_model_b_usage += 1

                pred = pred_b

                if pred_score_b > pred_score_a:
                    pred_ps = pred_b

            max_p_correct += pred == true
            max_p_correct_ps += pred_ps == true




        for true, pred_a, pred_b, pred_score_a, pred_score_b in zip(df_label_b['true'], df_label_b['classification_a'], df_label_b['classification_b'], df_label_b['max_prob_a'], df_label_b['max_prob_b']):

            pred = pred_b
            pred_ps = pred_b

            if pred_score_b < thr:
                max_p_model_b_usage_rev += 1

                pred = pred_a

                if pred_score_a > pred_score_b:
                    pred_ps = pred_a

            max_p_correct_rev += pred == true
            max_p_correct_ps_rev += pred_ps == true


        acc.append({
            'threshold' : thr,
            'correct' : max_p_correct,
            'model_b_used': max_p_model_b_usage,
            'label': label
        })

        acc_ps.append({
            'threshold' : thr,
            'correct' : max_p_correct_ps,
            'model_b_used': max_p_model_b_usage,
            'label': label
        })

        acc_rev.append({
            'threshold' : thr,
            'correct' : max_p_correct_rev,
            'model_b_used': max_p_model_b_usage_rev,
            'label': label
        })

        acc_ps_rev.append({
            'threshold' : thr,
            'correct' : max_p_correct_ps_rev,
            'model_b_used': max_p_model_b_usage_rev,
            'label': label
        })
    
    print(f'Threshold search for label {label} finished.')
    results_arr[label] = acc
    results_arr_ps[label] = acc_ps
    results_arr_rev[label] = acc_rev
    results_arr_ps_rev[label] = acc_ps_rev


def heuristic_search(trues, preds_a, preds_b, preds_score_a, preds_score_b, param_range, reverse, results, key_1, key_2):
    acc = []
    acc_ps = []
    # threshold_params = np.linspace(0.0001, 1, 1000)
    if not reverse:
        for threshold in tqdm(param_range, total=len(param_range)):            
            max_p_correct = 0
            max_p_correct_ps = 0
            max_p_model_b_usage = 0
            for true, pred_a, pred_b, pred_score_a, pred_score_b in zip(trues, preds_a, preds_b, preds_score_a, preds_score_b):
                if pred_score_a >= threshold:
                    max_p_correct += pred_a == true
                    max_p_correct_ps += pred_a == true
                else:
                    max_p_model_b_usage += 1
                    max_p_correct += pred_b == true
                    if pred_score_b > pred_score_a:
                        max_p_correct_ps += pred_b == true
                    else:
                        max_p_correct_ps += pred_a == true

            # print( max_p_correct / len(trues) * 100)
            acc.append({
                'threshold' : threshold,
                'accuracy' : max_p_correct,
                'model_b_usage': max_p_model_b_usage
            })

            acc_ps.append({
                'threshold' : threshold,
                'accuracy' : max_p_correct_ps,
                'model_b_usage': max_p_model_b_usage,
            })
    else:
        for threshold in tqdm(param_range, total=len(param_range)):
            max_p_correct = 0
            max_p_correct_ps = 0
            max_p_model_b_usage = 0
            for true, pred_a, pred_b, pred_score_a, pred_score_b in zip(trues, preds_a, preds_b, preds_score_a, preds_score_b):
                if pred_score_a <= threshold:
                    max_p_correct += pred_a == true
                    max_p_correct_ps += pred_a == true
                else:
                    max_p_model_b_usage += 1
                    max_p_correct += pred_b == true
                    if pred_score_b < pred_score_a:
                        max_p_correct_ps += pred_b == true
                    else:
                        max_p_correct_ps += pred_a == true
        
            acc.append({
                'threshold' : threshold,
                'accuracy' : max_p_correct,
                'model_b_usage': max_p_model_b_usage
            })

            acc_ps.append({
                'threshold' : threshold,
                'accuracy' : max_p_correct_ps,
                'model_b_usage': max_p_model_b_usage,
            })

    results[key_1] = acc
    results[key_2] = acc_ps


## the accuracy needs to be calculated on split length not whole dataset length
def heuristic_search_process(trues, preds_a, preds_b, preds_score_a, preds_score_b, param_range, reverse):
    acc = []
    acc_ps = []
    # threshold_params = np.linspace(0.0001, 1, 1000)
    if not reverse:
        for threshold in param_range:            
            max_p_correct = 0
            max_p_correct_ps = 0
            max_p_model_b_usage = 0
            for true, pred_a, pred_b, pred_score_a, pred_score_b in zip(trues, preds_a, preds_b, preds_score_a, preds_score_b):
                if pred_score_a >= threshold:
                    max_p_correct += pred_a == true
                    max_p_correct_ps += pred_a == true
                else:
                    max_p_model_b_usage += 1
                    max_p_correct += pred_b == true
                    if pred_score_b > pred_score_a:
                        max_p_correct_ps += pred_b == true
                    else:
                        max_p_correct_ps += pred_a == true

            # print('a', max_p_correct)
            acc.append({
                'threshold' : threshold,
                'accuracy' : max_p_correct,
                'model_b_usage': max_p_model_b_usage
            })

            acc_ps.append({
                'threshold' : threshold,
                'accuracy' : max_p_correct_ps,
                'model_b_usage': max_p_model_b_usage,
            })
    else:
        for threshold in param_range:
            max_p_correct = 0
            max_p_correct_ps = 0
            max_p_model_b_usage = 0
            for true, pred_a, pred_b, pred_score_a, pred_score_b in zip(trues, preds_a, preds_b, preds_score_a, preds_score_b):
                if pred_score_a <= threshold:
                    max_p_correct += pred_a == true
                    max_p_correct_ps += pred_a == true
                else:
                    max_p_model_b_usage += 1
                    max_p_correct += pred_b == true
                    if pred_score_b < pred_score_a:
                        max_p_correct_ps += pred_b == true
                    else:
                        max_p_correct_ps += pred_a == true
        
            acc.append({
                'threshold' : threshold,
                'accuracy' : max_p_correct,
                'model_b_usage': max_p_model_b_usage
            })

            acc_ps.append({
                'threshold' : threshold,
                'accuracy' : max_p_correct_ps,
                'model_b_usage': max_p_model_b_usage,
            })
    return (acc, acc_ps)

## the accuracy needs to be calculated on split length not whole dataset length
def heuristic_search_thread(args):
    trues, preds_a, preds_b, preds_score_a, preds_score_b, param_range, reverse = args[0], args[1], args[2], args[3], args[4], args[5], args[6]



    acc = []
    acc_ps = []
    # threshold_params = np.linspace(0.0001, 1, 1000)
    if not reverse:
        for threshold in param_range:            
            max_p_correct = 0
            max_p_correct_ps = 0
            max_p_model_b_usage = 0
            for true, pred_a, pred_b, pred_score_a, pred_score_b in zip(trues, preds_a, preds_b, preds_score_a, preds_score_b):
                if pred_score_a >= threshold:
                    max_p_correct += pred_a == true
                    max_p_correct_ps += pred_a == true
                else:
                    max_p_model_b_usage += 1
                    max_p_correct += pred_b == true
                    if pred_score_b > pred_score_a:
                        max_p_correct_ps += pred_b == true
                    else:
                        max_p_correct_ps += pred_a == true

            # print('a', max_p_correct)
            acc.append({
                'threshold' : threshold,
                'accuracy' : max_p_correct,
                'model_b_usage': max_p_model_b_usage
            })

            acc_ps.append({
                'threshold' : threshold,
                'accuracy' : max_p_correct_ps,
                'model_b_usage': max_p_model_b_usage,
            })
    else:
        for threshold in param_range:
            max_p_correct = 0
            max_p_correct_ps = 0
            max_p_model_b_usage = 0
            for true, pred_a, pred_b, pred_score_a, pred_score_b in zip(trues, preds_a, preds_b, preds_score_a, preds_score_b):
                if pred_score_a <= threshold:
                    max_p_correct += pred_a == true
                    max_p_correct_ps += pred_a == true
                else:
                    max_p_model_b_usage += 1
                    max_p_correct += pred_b == true
                    if pred_score_b < pred_score_a:
                        max_p_correct_ps += pred_b == true
                    else:
                        max_p_correct_ps += pred_a == true
        
            acc.append({
                'threshold' : threshold,
                'accuracy' : max_p_correct,
                'model_b_usage': max_p_model_b_usage
            })

            acc_ps.append({
                'threshold' : threshold,
                'accuracy' : max_p_correct_ps,
                'model_b_usage': max_p_model_b_usage,
            })

    return (acc, acc_ps)