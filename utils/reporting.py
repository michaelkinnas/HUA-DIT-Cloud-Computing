from sklearn.metrics import classification_report

def run_methodology_with_per_class_thresholds_and_calculate_classification_report(true, pred_a, pred_b, pred_score_a, pred_score_b, thresholds, rev=False):
    predictions = []
    model_b_used = 0

    if not rev:
        for pred_a, pred_b, pred_score_a, pred_score_b in zip(pred_a, pred_b, pred_score_a, pred_score_b):
            if pred_score_a >= thresholds[pred_a]:
                predictions.append(pred_a)
            else:
                model_b_used += 1
                predictions.append(pred_b)

    else:
         for pred_a, pred_b, pred_score_a, pred_score_b in zip(pred_a, pred_b, pred_score_a, pred_score_b):
            if pred_score_a <= thresholds[pred_a]:
                predictions.append(pred_a)
            else:
                model_b_used += 1
                predictions.append(pred_b)

    print(f"Model B usage {model_b_used / len(true) * 100:.2f}%")
    print(classification_report(true, predictions, digits = 4, zero_division=0))


def run_methodology_with_per_class_thresholds_with_postcheck_and_calculate_classification_report(true, pred_a, pred_b, pred_score_a, pred_score_b, thresholds, rev=False):
    predictions_ps = []
    model_b_used = 0
    if not rev:
        for pred_a, pred_b, pred_score_a, pred_score_b in zip(pred_a, pred_b, pred_score_a, pred_score_b):
            if pred_score_a >= thresholds[pred_a]:
                predictions_ps.append(pred_a)
            else:
                model_b_used += 1
                if pred_score_b > pred_score_a:
                    predictions_ps.append(pred_b)
                else:
                    predictions_ps.append(pred_a)
    else:
        for pred_a, pred_b, pred_score_a, pred_score_b in zip(pred_a, pred_b, pred_score_a, pred_score_b):
            if pred_score_a <= thresholds[pred_b]:
                predictions_ps.append(pred_a)
            else:
                model_b_used += 1
                if pred_score_b < pred_score_a:
                    predictions_ps.append(pred_b)
                else:                    
                    predictions_ps.append(pred_a)
    
    
    print(f"Model B usage {model_b_used / len(true) * 100:.2f}%")
    print(classification_report(true, predictions_ps, digits = 4, zero_division=0))



def run_methodology_and_calculate_classification_report(true, pred_a, pred_b, pred_score_a, pred_score_b, threshold, rev=False):
    predictions = []
    model_b_used = 0
    if not rev:
        for pred_a, pred_b, pred_score_a, pred_score_b in zip(pred_a, pred_b, pred_score_a, pred_score_b):
            if pred_score_a >= threshold:
                predictions.append(pred_a)
            else:
                model_b_used += 1
                predictions.append(pred_b)

    else:
         for pred_a, pred_b, pred_score_a, pred_score_b in zip(pred_a, pred_b, pred_score_a, pred_score_b):
            if pred_score_a <= threshold:
                predictions.append(pred_a)
            else:
                model_b_used += 1
                predictions.append(pred_b)

    print(f"Threshold {threshold:.4f}")
    print(f"Model B usage {model_b_used / len(true) * 100:.2f}%")
    print(classification_report(true, predictions, digits = 4, zero_division=0))



def run_methodology_with_postcheck_and_calculate_classification_report(true, pred_a, pred_b, pred_score_a, pred_score_b, threshold, rev=False):
    predictions_ps = []
    model_b_used = 0

    if not rev:
        for pred_a, pred_b, pred_score_a, pred_score_b in zip(pred_a, pred_b, pred_score_a, pred_score_b):
            if pred_score_a >= threshold:
                predictions_ps.append(pred_a)
            else:
                model_b_used += 1
                if pred_score_b > pred_score_a:
                    predictions_ps.append(pred_b)
                else:
                    predictions_ps.append(pred_a)
    else:
        for pred_a, pred_b, pred_score_a, pred_score_b in zip(pred_a, pred_b, pred_score_a, pred_score_b):
            if pred_score_a <= threshold:
                predictions_ps.append(pred_a)
            else:
                model_b_used += 1
                if pred_score_b < pred_score_a:
                    predictions_ps.append(pred_b)
                else:
                    predictions_ps.append(pred_a)
    
    print(f"Threshold {threshold:.4f}")
    print(f"Model B usage {model_b_used / len(true) * 100:.2f}%")
    print(classification_report(true, predictions_ps, digits = 4, zero_division=0))



def run_methodology_and_calculate_classification_report2(true, pred_a, pred_b, pred_score_a, pred_score_b, threshold, name_a, name_b):
    predictions = []
    model_b_used = 0

    if pred_score_a.name == 'entropy':
        for pred_a, pred_b, pred_score_a, pred_score_b in zip(pred_a, pred_b, pred_score_a, pred_score_b):
            if pred_score_a <= threshold:
                predictions.append(pred_a)
            else:
                model_b_used += 1
                predictions.append(pred_b)
    else:
        for pred_a, pred_b, pred_score_a, pred_score_b in zip(pred_a, pred_b, pred_score_a, pred_score_b):
            if pred_score_a >= threshold:
                predictions.append(pred_a)
            else:
                model_b_used += 1
                predictions.append(pred_b)

    # dict = classification_report(true, predictions, digits = 4, zero_division=0, output_dict=True)
    # print(f"A: {name_a}, \nB: {name_b}")    
    # print(f"Threshold {threshold:.4f} | B model usage {model_b_used / len(true) * 100:.2f}%")
    # print(f"Accuracy: {dict['accuracy']:.4f}\nPrecision: {dict['macro avg']['precision']:.4f}\nRecall: {dict['macro avg']['recall']:.4f}\nF1: {dict['macro avg']['f1-score']:.4f}")

    return classification_report(true, predictions, digits = 4, zero_division=0, output_dict=True), threshold, model_b_used / len(true)



def run_methodology_with_postcheck_and_calculate_classification_report2(true, pred_a, pred_b, pred_score_a, pred_score_b, threshold, name_a, name_b):
    predictions_ps = []
    model_b_used = 0

    if pred_score_a.name == 'entropy':
        for pred_a, pred_b, pred_score_a, pred_score_b in zip(pred_a, pred_b, pred_score_a, pred_score_b):
            if pred_score_a <= threshold:
                predictions_ps.append(pred_a)
            else:
                model_b_used += 1
                if pred_score_b < pred_score_a:
                    predictions_ps.append(pred_b)
                else:
                    predictions_ps.append(pred_a)
    else:
        for pred_a, pred_b, pred_score_a, pred_score_b in zip(pred_a, pred_b, pred_score_a, pred_score_b):
            if pred_score_a >= threshold:
                predictions_ps.append(pred_a)
            else:
                model_b_used += 1
                if pred_score_b > pred_score_a:
                    predictions_ps.append(pred_b)
                else:
                    predictions_ps.append(pred_a)

    # dict = classification_report(true, predictions_ps, digits = 4, zero_division=0, output_dict=True)
    # print(f"A: {name_a}, \nB: {name_b}")    
    # print(f"Threshold {threshold:.4f} | B model usage {model_b_used / len(true) * 100:.2f}%")
    # print(f"Accuracy: {dict['accuracy']:.4f}\nPrecision: {dict['macro avg']['precision']:.4f}\nRecall: {dict['macro avg']['recall']:.4f}\nF1: {dict['macro avg']['f1-score']:.4f}")
    return classification_report(true, predictions_ps, digits = 4, zero_division=0, output_dict=True), threshold, model_b_used / len(true)

def run_methodology_with_oracle_and_calculate_classification_report(true, pred_a, pred_b):
    predictions = []
    model_b_used = 0


    for true_s, pred_a, pred_b, in zip(true, pred_a, pred_b):
        if pred_a == true_s:
            predictions.append(pred_a)
        else:
            model_b_used += 1
            predictions.append(pred_b)
            
    print(f"Model B usage {model_b_used / len(true) * 100:.2f}%")
    print(classification_report(true, predictions, digits = 4, zero_division=0))