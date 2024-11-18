from pathlib import Path
import re
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd


def test_accuracy(model, dataloader, device, loss_fn=torch.nn.CrossEntropyLoss()):
    model.eval()
    accuracy = 0.0
    loss = 0.0
    total = 0.0

    y_pred = []
    y_true = []
    
    with torch.no_grad():
        test_results = pd.DataFrame()
        for (images, labels, metadata) in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            # run the model on the test set to predict labels
            outputs = model(images)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            loss += loss_fn(outputs, labels)
            # the label with the highest energy will be our prediction
            probability, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

            y_pred.extend(predicted.data.cpu().numpy())
            y_true.extend(labels.data.cpu().numpy())

            data = pd.concat(
                (
                    pd.DataFrame.from_dict(metadata), 
                    pd.DataFrame.from_dict({
                        'probability': probability.data.cpu().numpy(),
                        'predicted': predicted.data.cpu().numpy(),
                        'ground_truth': labels.data.cpu().numpy()
                        })
                ),
                axis=1
            )
            test_results = pd.concat((test_results, data), axis=0, ignore_index=True)

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    loss = loss.item() / total

    # confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')

    return accuracy, loss, cf_matrix, test_results


def train_model(model, dataloaders, loss_fn_train, loss_fn_eval, optimizer, scheduler, device, n_epochs, batch_size, exp_id):
    best_val_loss = np.inf
    model.to(device)
    train_loss_each_epoch = []
    val_loss_each_epoch = []
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        n_batches = 0
        running_loss = 0.0
        running_accuracy = 0.0
        total=0.0

        model.train()
        train_df = pd.DataFrame()
        with tqdm(dataloaders['train'], unit="batch", colour='green') as tepoch:
            for (images, labels, metadata) in tepoch:
                tepoch.set_description(f"Experiment {exp_id} | Epoch {epoch}/{n_epochs-1}")
                n_batches += 1
                # get the inputs
                images = images.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # predict classes using images from the training set
                outputs = model(images)

                # compute accuracy
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                probability, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                running_accuracy += (predicted == labels).sum().item()

                # compute the loss based on model output and real labels
                loss = loss_fn_train(outputs, labels)
                tepoch.set_postfix(loss=loss.item() / batch_size)

                # backpropagate the loss
                loss.backward()
                # adjust parameters based on the calculated gradients
                optimizer.step()

                # Update the running loss
                running_loss += loss.item()

                # Track predictions in df
                data = pd.concat(
                    (
                        pd.DataFrame.from_dict(metadata), 
                        pd.DataFrame.from_dict({
                            'probability': probability.data.cpu().numpy(),
                            'predicted': predicted.data.cpu().numpy(),
                            'ground_truth': labels.data.cpu().numpy()
                            })
                    ),
                    axis=1
                )
                train_df = pd.concat((train_df, data), axis=0, ignore_index=True)
        train_df = pd.concat(
            (train_df, pd.Series(['train']*len(train_df), name='dataset_split')),
            axis=1,
            )

        train_accuracy = 100 * running_accuracy / total
        train_loss = running_loss / total
        train_loss_each_epoch.append(train_loss)
        
        print(f' Experiment {exp_id} | Epoch {epoch}/{n_epochs-1} | Training epoch loss: {train_loss:.5f} | Training epoch accuracy {train_accuracy: .2f}')
        # zero the loss
        running_loss = 0.0

        # Compute and print the average validation metrics of this epoch
        val_accuracy, val_loss, cf_matrix, val_df = test_accuracy(model, dataloaders['val'], device, loss_fn_eval)
        val_df = pd.concat(
            (val_df, pd.Series(['val']*len(val_df), name='dataset_split')),
            axis=1
            )
        val_loss_each_epoch.append(val_loss)
        tnr, _, _, tpr = 100*cf_matrix.ravel()
        
        print(
            f'Experiment {exp_id} | Epoch {epoch}/{n_epochs-1} |', 
            f'Validation set loss: {val_loss: .5f} | Validation set accuracy: {val_accuracy: .2f} | TNR: {tnr: .2f} | TPR: {tpr: .2f}'
            )
        
        # we want to save the model if the accuracy is the best
        if val_loss < best_val_loss:
            print("Saving new best model")
            best_val_epoch=epoch
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_val_model_state_dict = model.state_dict()
            best_val_optimizer_state_dict = optimizer.state_dict()
            best_val_scheduler_state_dict = scheduler.state_dict()
            best_val_df = val_df
            best_train_df = train_df

        # Step LR scheduler
        scheduler.step(val_loss)

    train_results = {
        'final_model_state_dict': model.state_dict(),
        'final_optimizer_state_dict': optimizer.state_dict(),
        'final_scheduler_state_dict': scheduler.state_dict(),
        'train_loss_each_epoch': train_loss_each_epoch,
        'val_loss_each_epoch': val_loss_each_epoch,
        'best_val_epoch': best_val_epoch,
        'best_val_loss': best_val_loss,
        'best_val_accuracy': best_val_accuracy,
        'best_val_model_state_dict': best_val_model_state_dict,
        'best_val_optimizer_state_dict': best_val_optimizer_state_dict,
        'best_val_scheduler_state_dict': best_val_scheduler_state_dict,
        'best_val_df': best_val_df,
        'best_train_df': best_train_df,
        'final_val_df': val_df,
        'final_train_df': train_df,
        'final_val_accuracy': val_accuracy
    }        
    return train_results