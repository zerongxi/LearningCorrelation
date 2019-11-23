import numpy as np
import data_utils
import sklearn, sklearn.model_selection, sklearn.neighbors, sklearn.multiclass, sklearn.linear_model
from sklearn.preprocessing import normalize

def gravity_embedding(correlation_matrix, n_dims, alpha=0.5, stop_criterion=1e-5, max_steps=10**4):
    embedding = -1 + 2 * np.random.random((correlation_matrix.shape[0], n_dims)).astype(np.float32)

    diff = 1.0
    for step in range(max_steps):
        if diff < stop_criterion:
            break

        updated = correlation_matrix.dot(embedding)
        updated *= alpha
        updated += embedding * (1 - alpha)
        updated -= np.mean(updated, axis=0)
        updated /= np.std(updated, axis=0)

        diff = np.mean(np.abs(updated - embedding))
        if step % 30 == 29:
            print("step: %5d, diff: %.3e, range: (%.3e, %.3e)" % (step+1, diff, np.min(updated), np.max(updated)))
        embedding = updated

    return embedding


def walk(correlation_matrix, n_steps):
    base = normalize(correlation_matrix, norm="l1", axis=0)
    no_loop_2 = correlation_matrix.dot(correlation_matrix) \
                * (1 - np.eye(correlation_matrix.shape[0]))
    no_loop_2 = normalize(no_loop_2, norm="l1", axis=0)
    weights = [base, no_loop_2]
    for i in range(n_steps - 2):
        weights.append(normalize(weights[-2].dot(no_loop_2), norm="l1", axis=0))
    ret = np.stack(weights, -1) * np.power(0.8, range(n_steps))
    ret = np.sum(ret, axis=-1)
    ret = normalize(ret, norm="l1", axis=0)
    return ret


mydata = data_utils.Data("./data/BlogCatalog3/")
correlation_matrix = walk(mydata.correlation_matrix, 5)
print(np.mean(correlation_matrix))
embeddings = gravity_embedding(correlation_matrix, 4, alpha=0.5, stop_criterion=2e-6)

trainX, testX, trainY, testY = sklearn.model_selection.train_test_split(
    embeddings, mydata.index2group, test_size=0.9
)


def eval(model):
    model.fit(trainX, trainY)
    # print("accuracy: ", model.score(testX, testY))
    pred = model.predict(testX)
    print("micro f1: ", sklearn.metrics.f1_score(testY, pred, labels=np.unique(mydata.index2group), average="micro"))
    print("macro f1: ", sklearn.metrics.f1_score(testY, pred, labels=np.unique(mydata.index2group), average="macro"))


knn = sklearn.neighbors.KNeighborsClassifier()
knn_ovr = sklearn.multiclass.OneVsRestClassifier(sklearn.neighbors.KNeighborsClassifier(n_neighbors=10), 3)
lr_ovr = sklearn.multiclass.OneVsRestClassifier(sklearn.linear_model.LogisticRegression(solver="lbfgs"), 3)
eval(knn)
eval(knn_ovr)
eval(lr_ovr)