import matplotlib.pyplot as plt
from io import BytesIO
import base64

from matplotlib.figure import Figure


def classification_result(classes, preds):
    # plt.bar(classes, preds)

    fig = Figure()
    ax = fig.subplots()
    # ax.plot([1, 2])
    ax.bar(classes, preds)
    ax.set_title('CIFAR10 Classifier')
    ax.set_ylabel('Probability')
    ax.set_xlabel('Classes')

    buf = BytesIO()
    fig.savefig(buf, format="png")

    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return data
    # return f"<img src='data:image/png;base64,{data}'/>"

    # plt.savefig(buffer, format='png')
    # PNG = buffer.getvalue()
    # return PNG