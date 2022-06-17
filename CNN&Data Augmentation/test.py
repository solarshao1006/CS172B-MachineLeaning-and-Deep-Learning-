import torch
def test(device, test_loader, model):
    cnn.eval()
    total_test_label = []
    total_test_predictions = []

    print("Start Testing")
    with torch.no_grad():
      for idx, (img, label) in enumerate(test_loader):
        img = Variable(img).to(device)
        label = Variable(label).type(torch.LongTensor).to(device)
        scores = cnn(img)
        loss = criterion(scores, label)
        
        _, predictions = scores.max(dim=1)
        total_test_predictions += predictions.cpu()
        total_test_label += label.cpu()
        
    total_test_acc = metrics.accuracy_score(total_test_label, total_test_predictions)

    print('Accuracy: {:.6f}'.format(total_test_acc))
    return total_test_acc
