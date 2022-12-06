import mnist
from Model import Model

train_images = mnist.train_images()[:2000]
train_labels = mnist.train_labels()[:2000]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

m = Model()
loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i % 100 == 99:
        print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (i + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0

    im = im.reshape(28, 28, 1)
    l, acc = m.train(im, label, 0.005)
    loss += l
    num_correct += acc

loss = 0
num_correct = 0

for im, label in zip(test_images, test_labels):
    im = im.reshape(28, 28, 1)
    _, l, acc = m.forward(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)