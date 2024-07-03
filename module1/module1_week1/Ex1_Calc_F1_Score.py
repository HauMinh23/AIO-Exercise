# câu 1: Viết function thực hiện đánh giá classification model bằng F1-Score.
import math


def calc_f1_score(tp, fp, fn):
    # tính Precision
    precision = tp/(tp+fp)

    # tính recall
    recall = tp/(tp+fn)

    # tính f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)

    # kiểm tra giá trị nhận vào là type int hay không
    if isinstance(tp, int):
        print('tp must be int')
        return
    if isinstance(fp, int):
        print('fp must be int')
        return
    if isinstance(fn, int):
        print('fn must be int')
        return
    # kiểm tra giá trị nhận vào có lớn hơn không hay không
    if (tp <= 0) or (fp <= 0) or (fn <= 0):
        print('tp and fp and fn must be greater than zero')
        return
    return f1_score


assert math.isclose((calc_f1_score(1, 1, 1)), 0.5,
                    rel_tol=1e-09, abs_tol=1e-09)
(calc_f1_score(1, 3, 5))
