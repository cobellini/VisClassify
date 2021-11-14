from matplotlib import pyplot as plt


class ClassifierEvaluation:
    def __init__(self, classifier_history):
        self.classifier_history = classifier_history

    def classifier_accuracy(self):
        fig, axes = plt.subplots(2)
        #accuracy subplot
        axes[0].plot(self.classifier_history.history['accuracy'], label='training accuracy')
        axes[0].plot(self.classifier_history.history['val_accuracy'], label='validation accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(loc='lower right')
        axes[0].set_title('Accuracy Evaluation')
        plt.savefig('../figs/dissertation_accuracy.png')

    def classifier_error(self):
        fig, axes = plt.subplots(2)
        axes[1].plot(self.classifier_history.history['loss'], label='training error')
        axes[1].plot(self.classifier_history.history['val_loss'], label='validation error')
        axes[1].set_ylabel('Error')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(loc='upper right')
        axes[1].set_title('Error Evaluation')
        plt.savefig('../figs/disstertation_error.png')