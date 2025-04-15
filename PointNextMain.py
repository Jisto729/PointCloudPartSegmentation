from ShapeNetPartDataset import ShapeNetPartDataset
from Network import Network

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

#TODO remove when working
def visualize_activations(model, PC, label, layer_name, epoch):
    activations = {}

    # Hook function to capture activations
    def hook(module, input, output):
        if isinstance(output, tuple):
            activations[layer_name] = output[1].detach().cpu().numpy()  # Extract only the feature tensor
        else:
            activations[layer_name] = output.detach().cpu().numpy()

    # Get the layer by name
    for name, _ in network.named_modules():
        print(name)
    layer = dict([*model.named_modules()])[layer_name]
    # layer = model
    # for subname in layer_name.split('.'):
    #     layer = getattr(layer, subname)
    hook_handle = layer.register_forward_hook(hook)

    # Run forward pass
    with torch.no_grad():
        _ = model(PC, label)

        # Remove the hook
    hook_handle.remove()

    # Extract activations
    act = activations[layer_name]

    # Plot activations for visualization
    plt.figure(figsize=(10, 4))
    plt.imshow(act[0], aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(f"Activations in {layer_name} at Epoch {epoch}")
    plt.show()


def visualize_class_predictions(points, predicted_classes, num_classes=50):
    colors = np.random.rand(num_classes, 3)  # Generate unique colors for each class
    point_colors = colors[predicted_classes]  # Assign color to each point

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    #setting to full screen
    o3d.visualization.draw_geometries([pcd], window_name='Class Predictions', width=1920, height=1080, left=0, top=0)


########################################################################################################################


import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
training_dataset = ShapeNetPartDataset('D:\\bak\\datasets\\shapeNetPart\\shapenetpart_hdf5_2048\\shapenetpart_hdf5_2048', 'train_debug')
validation_dataset = ShapeNetPartDataset('D:\\bak\\datasets\\shapeNetPart\\shapenetpart_hdf5_2048\\shapenetpart_hdf5_2048', 'val_debug')
testing_dataset = ShapeNetPartDataset('D:\\bak\\datasets\\shapeNetPart\\shapenetpart_hdf5_2048\\shapenetpart_hdf5_2048', 'test_debug')

epochs = 1
batch_size = 20

dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
from collections import Counter

#counting the number of classes so they can be weighted properly
class_counts = Counter()

for _, label, _ in training_dataset:
    if len(label.shape) > 0 and label.shape[0] > 1:
        for single_label in label:
            class_counts[single_label.item()] += 1
    else:
        # If label is already scalar-like
        class_counts[label.item()] += 1

total_samples = sum(class_counts.values())
class_weights = {class_idx: total_samples / count for class_idx, count in class_counts.items()}

max_class_index = training_dataset.get_part_classes() - 1
weights = [class_weights.get(i, 1.0) for i in range(max_class_index + 1)]
weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)


network = Network(batch_size, training_dataset.get_part_classes(), device, training_dataset.get_object_classes())
print("created network")
network.to(device)
print(network)
network.apply(weights_init)

loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

finalPC = torch.tensor([])
finalLabel = torch.tensor([])
finalClassLabel = torch.tensor([])

weight_changes = {}
activations = {}

for epoch in range(epochs):
    processed_clouds = 0
    print(f'Starting epoch {epoch}')
    # TRAINING
    #Gk
    total_per_class_ground_truth_labels = torch.zeros(training_dataset.get_part_classes()).to(device)
    #Pk
    total_per_class_correctly_predicted_labels = torch.zeros(training_dataset.get_part_classes()).to(device)
    #Nk
    total_per_class_predicted_labels = torch.zeros(training_dataset.get_part_classes()).to(device)

    total_loss = 0.0

    network.train()
    for point_cloud, point_labels, class_label in dataloader:
        print("proccessed clouds:", processed_clouds)
        processed_clouds += batch_size
        point_cloud = point_cloud.to(device)
        finalPC = point_cloud
        point_labels = point_labels.to(device)
        finalLabel = point_labels
        class_label = class_label.to(device)
        #one hot encoding the classLabel
        class_label = torch.nn.functional.one_hot(class_label, num_classes=training_dataset.get_object_classes())
        finalClassLabel = class_label

        optimizer.zero_grad()
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                with_flops=True
        ) as prof_train:
            prediction, _ = network(point_cloud.permute(0, 2, 1), class_label.permute(0, 2, 1))
        prediction = prediction.permute(0,2,1)

        #computing accuracies and ious
        total_per_class_ground_truth_labels += torch.bincount(point_labels.flatten(), minlength=training_dataset.get_part_classes())

        predicted_labels = torch.argmax(prediction, dim=-1)
        total_per_class_predicted_labels += torch.bincount(predicted_labels.flatten(), minlength=training_dataset.get_part_classes())

        correct_pred_mask = predicted_labels == point_labels

        correct_labels = point_labels.flatten()[correct_pred_mask.flatten()]
        total_per_class_correctly_predicted_labels += torch.bincount(correct_labels.flatten(), minlength=training_dataset.get_part_classes())

        prediction=prediction.permute(0,2,1)
        loss = loss_fn(prediction, point_labels)
        loss.backward()
        total_loss += loss.item()

        old_weights = {name: param.clone().detach() for name, param in network.named_parameters() if
                        param.requires_grad}
        optimizer.step()


    overall_accuracy = total_per_class_correctly_predicted_labels.sum().item() / total_per_class_ground_truth_labels.sum().item()

    present_class_mask = total_per_class_ground_truth_labels > 0
    per_class_overall_accuracy = total_per_class_correctly_predicted_labels[present_class_mask].float() / total_per_class_ground_truth_labels[present_class_mask].float()
    mean_accuracy = per_class_overall_accuracy.mean().item()

    per_class_iou = (total_per_class_correctly_predicted_labels[present_class_mask].float()/
                     (total_per_class_ground_truth_labels[present_class_mask].float() +
                      total_per_class_predicted_labels[present_class_mask].float() -
                      total_per_class_correctly_predicted_labels[present_class_mask].float()))
    result_miou = per_class_iou.mean().item()

    print(f'Finished epoch {epoch} training')
    print("Loss:", total_loss)
    print("Overall Accuracy:", overall_accuracy)
    print("Mean Accuracy:", mean_accuracy)
    print("Miou:", result_miou)

    #VALIDATION
    # Gk
    total_per_class_ground_truth_labels = torch.zeros(training_dataset.get_part_classes()).to(device)
    # Pk
    total_per_class_correctly_predicted_labels = torch.zeros(training_dataset.get_part_classes()).to(device)
    # Nk
    total_per_class_predicted_labels = torch.zeros(training_dataset.get_part_classes()).to(device)

    network.eval()
    val_total_loss = 0.0

    with torch.no_grad():
        for point_cloud, point_labels, class_label in val_dataloader:
            point_cloud = point_cloud.to(device)
            point_labels = point_labels.to(device)
            class_label = class_label.to(device)
            #one hot encoding the classLabel
            class_label = torch.nn.functional.one_hot(class_label, num_classes=validation_dataset.get_object_classes())

            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_stack=True,
                    profile_memory=True,
                    with_flops=True
            ) as prof_val:
                prediction, _ = network(point_cloud.permute(0, 2, 1), class_label.permute(0, 2, 1))
            prediction = prediction.permute(0, 2, 1)

            total_per_class_ground_truth_labels += torch.bincount(point_labels.flatten(),
                                                                  minlength=validation_dataset.get_part_classes())

            predicted_labels = torch.argmax(prediction, dim=-1)
            total_per_class_predicted_labels += torch.bincount(predicted_labels.flatten(),
                                                               minlength=training_dataset.get_part_classes())
            correct_pred_mask = predicted_labels == point_labels

            correct_labels = point_labels.flatten()[correct_pred_mask.flatten()]
            total_per_class_correctly_predicted_labels += torch.bincount(correct_labels.flatten(),
                                                                         minlength=training_dataset.get_part_classes())

            prediction = prediction.permute(0, 2, 1)

            loss = loss_fn(prediction, point_labels).item()
            val_total_loss += loss

    overall_accuracy = total_per_class_correctly_predicted_labels.sum().item() / total_per_class_ground_truth_labels.sum().item()

    present_class_mask = total_per_class_ground_truth_labels > 0
    per_class_overall_accuracy = total_per_class_correctly_predicted_labels[present_class_mask].float() / \
                                 total_per_class_ground_truth_labels[present_class_mask].float()
    mean_accuracy = per_class_overall_accuracy.mean().item()

    per_class_iou = (total_per_class_correctly_predicted_labels[present_class_mask].float() /
                     (total_per_class_ground_truth_labels[present_class_mask].float() +
                      total_per_class_predicted_labels[present_class_mask].float() -
                      total_per_class_correctly_predicted_labels[present_class_mask].float()))
    result_miou = per_class_iou.mean().item()

    print(f'Finished epoch {epoch} validation')
    print("Loss:", total_loss)
    print("Overall Accuracy:", overall_accuracy)
    print("Mean Accuracy:", mean_accuracy)
    print("Miou:", result_miou)

    #every 5 epochs visualize activations
    if epoch % 5 == 0:
        visualize_activations(network, point_cloud.permute(0, 2, 1),class_label.permute(0, 2, 1), "FP_layer1", epoch)

    ##REPLACE
    for name, param in network.named_parameters():
        if param.requires_grad:
            weight_change = (param - old_weights[name]).norm().item()
            if name not in weight_changes:
                weight_changes[name] = []
            weight_changes[name].append(weight_change)


for name, updates in weight_changes.items():
    plt.plot(updates, label=name)

plt.xlabel("Epochs")
plt.ylabel("Weight Change Norm")
plt.yscale("log")
plt.show()

#visualizing some predictions
network.eval()
with torch.no_grad():
    prediction, learned_features = network(finalPC.permute(0,2,1), finalClassLabel.permute(0,2,1))

prediction = prediction.permute(0, 2, 1)
prediction = torch.nn.functional.softmax(prediction, dim=2)

predicted_classes = torch.argmax(prediction, dim=2)

for i in range(min(15, batch_size)):
    visualize_class_predictions(finalPC[i].cpu().numpy(), finalLabel[i].cpu().numpy())
    visualize_class_predictions(finalPC[i].cpu().numpy(), predicted_classes[i].cpu().numpy())

#TESTING
network.eval()
# Gk
total_per_class_ground_truth_labels = torch.zeros(training_dataset.get_part_classes()).to(device)
# Pk
total_per_class_correctly_predicted_labels = torch.zeros(training_dataset.get_part_classes()).to(device)
# Nk
total_per_class_predicted_labels = torch.zeros(training_dataset.get_part_classes()).to(device)
test_total_loss = 0.0

with torch.no_grad():
    for point_cloud, point_labels, class_label in test_dataloader:
        point_cloud = point_cloud.to(device)
        point_labels = point_labels.to(device)
        class_label = class_label.to(device)
        #one hot encoding the classLabel
        class_label = torch.nn.functional.one_hot(class_label, num_classes=training_dataset.get_object_classes())
        prediction, _ = network(point_cloud.permute(0, 2, 1), class_label.permute(0, 2, 1))
        prediction = prediction.permute(0, 2, 1)

        total_per_class_ground_truth_labels += torch.bincount(point_labels.flatten(),
                                                              minlength=validation_dataset.get_part_classes())

        predicted_labels = torch.argmax(prediction, dim=-1)
        total_per_class_predicted_labels += torch.bincount(predicted_labels.flatten(),
                                                           minlength=training_dataset.get_part_classes())
        correct_pred_mask = predicted_labels == point_labels

        correct_labels = point_labels.flatten()[correct_pred_mask.flatten()]
        total_per_class_correctly_predicted_labels += torch.bincount(correct_labels.flatten(),
                                                                     minlength=training_dataset.get_part_classes())

        prediction = prediction.permute(0, 2, 1)
        loss = loss_fn(prediction, point_labels).item()
        test_total_loss += loss

overall_accuracy = total_per_class_correctly_predicted_labels.sum().item() / total_per_class_ground_truth_labels.sum().item()

present_class_mask = total_per_class_ground_truth_labels > 0
per_class_overall_accuracy = total_per_class_correctly_predicted_labels[present_class_mask].float() / \
                             total_per_class_ground_truth_labels[present_class_mask].float()
mean_accuracy = per_class_overall_accuracy.mean().item()

per_class_iou = (total_per_class_correctly_predicted_labels[present_class_mask].float() /
                 (total_per_class_ground_truth_labels[present_class_mask].float() +
                  total_per_class_predicted_labels[present_class_mask].float() -
                  total_per_class_correctly_predicted_labels[present_class_mask].float()))
result_miou = per_class_iou.mean().item()

print(f'Finished testing')
print("Loss:", test_total_loss)
print("Overall Accuracy:", overall_accuracy)
print("Mean Accuracy:", mean_accuracy)
print("Miou:", result_miou)

print(prof_train.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print(prof_val.key_averages().table(sort_by="cuda_time_total", row_limit=10))