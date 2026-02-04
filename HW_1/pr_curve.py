import numpy as np


def build_precision_recall_curve(
    true_labels: np.ndarray, predicted_probas: np.ndarray
) -> np.ndarray:

    """
    Данная функция строит PR-кривую для задачи бинарной классификации.
    В случае, когда нет ни одного объекта положительного класса функция должна вызывать ValueError().

    Args:
        true_labels (np.ndarray): Массив истинных меток класса. Состоит из 0 и 1.
            1 считается меткой положительного класса.
        predicted_probas (np.ndarray): Массив предсказанных вероятностей принадлежности объекта
            к положительному классу.

    Returns:
        np.ndarray: Массив размерами (len(true_labels)+1, 2), где в каждой строчке стоит пара (precision, recall), первым элементом всегда идет (0, 1)
    """
    total_pos = np.sum(true_labels)
    if total_pos == 0:
        raise ValueError("none positive labels")

    sorted_labels = true_labels[np.argsort(-predicted_probas)]
    pref_sum = np.cumsum(sorted_labels)

    pr_curve = np.empty((len(true_labels) + 1, 2))
    pr_curve[0] = (0, 1)

    for i in range(0, len(pref_sum)):
        tp = pref_sum[i]

        precision = tp / (i + 1)
        recall = tp / total_pos

        pr_curve[i + 1] = (recall, precision)

    return pr_curve
    pass

