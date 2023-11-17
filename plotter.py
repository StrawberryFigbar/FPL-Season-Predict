import matplotlib.pyplot as plt
import numpy as np


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(275))
    plt.yticks([])
    thisplot = plt.bar(range(275), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    plt.show()


def plot_predict(predictions_array, true_labels):
    correct = 0
    within_10 = 0
    within_25 = 0
    within_50 = 0
    incorrect = 0
    for i in range(len(predictions_array)):

        if (np.argmax(predictions_array[i])) == np.argmax(true_labels[i]):
            correct += 1
        elif abs(np.argmax(predictions_array[i]) - np.argmax(true_labels[i])) <= 10:
            within_10 += 1
        elif abs(np.argmax(predictions_array[i]) - np.argmax(true_labels[i])) <= 25:
            within_25 += 1
        elif abs(np.argmax(predictions_array[i]) - np.argmax(true_labels[i])) <= 50:
            within_50 += 1
        else:
            incorrect += 1

    categories = ['Correct', 'Within 10',
                  'Within 25', 'Within 50', ' Incorrect']
    counts = [correct, within_10, within_25, within_50, incorrect]

    plt.bar(categories, counts)
    plt.xlabel('Prediction Categories')
    plt.ylabel('Count')
    plt.title('Prediction Categories Distribution')
    plt.grid(True)
    plt.show()
