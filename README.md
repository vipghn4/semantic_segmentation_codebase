# Introduction

**Default configuration**:

* Data root: "/home/cotai/giang/datasets/VOC-2012"

* Label map: "/home/cotai/giang/semantic_segmentation/misc/label_map.json"

## 16/03/2020

**Test**: PASSED

* [x] StandardDataset ~> OK

* [x] Model ~> Bug (open-source UNet implementation is terrible)

**Problems**:

* Problems with VOC-2012 dataset:
    
    * Mean IoU is not a good way of evaluating a multi-class semantic segmentator

        * Explain: when class-imbalance, i.e. most pixels are background, and the model outputs predict all pixels as background, the IoU is still very high
    
    * VOC-2012 dataset is strongly imbalanced and difficult

* Other problems:

    * Current hardwares of COTAI is too slow to test codebase, i.e. have to train until convergence to see if things go right

**Solutions**:

* Find a small-and-easy dataset for quick experimenting

* Only focus on easy problems, not hard ones due to limtied hardware capacity

## 17/03/2020

**Notes**: https://github.com/tensorflow/models/blob/master/research/deeplab/train.py#L406

* Pipeline from https://github.com/tensorflow/models/blob/master/research/deeplab/train.py#L273
    
    1. Build dataset for patchwise training and whole-image inference, with augmentation, i.e. random scale >>> random crop >>> random vertical flip
    
    2. Build DeeplabV3 head with CELoss with hard example mining, class weights
    
    3. Get learning rate scheduler, including slow-start lr scheduler and ordinary scheduler
    
    4. Get optimizer, either momentum (in paper) or adam
    
    5. If transfer learning from classification task, multiple gradients of last layer(s) by some constant to enlarge the gradients

* Deeplab use multi-scale CELoss
    
    * Link: https://github.com/tensorflow/models/blob/28f6182fc9afaf11104a5abe7c21b57b6aeb30e2/research/deeplab/utils/train_utils.py#L33

    * Transform mask:
    
    ```
    scaled_labels = tf.image.resize_nearest_neighbor(
          labels,
          preprocess_utils.resolve_shape(logits, 4)[1:3],
          align_corners=True
    )
    scaled_labels = tf.reshape(scaled_labels, shape=[-1])
    
    weights = utils.get_label_weight_mask(scaled_labels, ignore_label, num_classes, label_weights=loss_weight)
    keep_mask = tf.cast(tf.not_equal(scaled_labels, ignore_label), dtype=tf.float32)
    
    if gt_is_matting_map:
        # When the groundtruth is integer label mask, we can assign class
        # dependent label weights to the loss. When the groundtruth is image
        # matting confidence, we do not apply class-dependent label weight (i.e.,
        # label_weight = 1.0).
        if loss_weight != 1.0:
            raise ValueError('loss_weight must equal to 1 if groundtruth is matting map.')

        # Assign label value 0 to ignore pixels. The exact label value of ignore
        # pixel does not matter, because those ignore_value pixel losses will be
        # multiplied to 0 weight.
        train_labels = scaled_labels * keep_mask

        train_labels = tf.expand_dims(train_labels, 1)
        train_labels = tf.concat([1 - train_labels, train_labels], axis=1)
    else:
        train_labels = tf.one_hot(caled_labels, num_classes, on_value=1.0, off_value=0.0)
    ```
    
    * Only focus on top-K hard pixels (hard example mining) for initial training steps and gradually sample less hard pixels (i.e. loss sampling):
    
    ```
    if top_k_percent_pixels == 1.0:
        total_loss = tf.reduce_sum(weighted_pixel_losses)
        num_present = tf.reduce_sum(keep_mask)
        loss = _div_maybe_zero(total_loss, num_present)
        tf.losses.add_loss(loss)
    else:
        num_pixels = tf.to_float(tf.shape(logits)[0])
        # Compute the top_k_percent pixels based on current training step.
        if hard_example_mining_step == 0:
            # Directly focus on the top_k pixels.
            top_k_pixels = tf.to_int32(top_k_percent_pixels * num_pixels)
        else:
            # Gradually reduce the mining percent to top_k_percent_pixels.
            global_step = tf.to_float(tf.train.get_or_create_global_step())
            ratio = tf.minimum(1.0, global_step / hard_example_mining_step)
            top_k_pixels = tf.to_int32((ratio * top_k_percent_pixels + (1.0 - ratio)) * num_pixels)
        top_k_losses, _ = tf.nn.top_k(weighted_pixel_losses,
                                    k=top_k_pixels,
                                    sorted=True,
                                    name='top_k_percent_pixels')
        total_loss = tf.reduce_sum(top_k_losses)
        num_present = tf.reduce_sum(
            tf.to_float(tf.not_equal(top_k_losses, 0.0)))
        loss = _div_maybe_zero(total_loss, num_present)
        tf.losses.add_loss(loss)
    ```

* DeeplabV3 first trained with SlowStartLRScheduler then PolyLRScheduler or some other LR Scheduler

    * Link: https://github.com/tensorflow/models/blob/28f6182fc9afaf11104a5abe7c21b57b6aeb30e2/research/deeplab/utils/train_utils.py#L268

* DeeplabV3 uses MomentumOptimizer or AdamOptimizer

* The gradient multipliers will adjust the learning rates for model variables. For the task of semantic segmentation, the models are usually fine-tuned from the models trained on the task of image classification. To fine-tune the models, we usually set larger (e.g., 10 times larger) learning rate for the parameters of last layer(s).

    * Pipeline: 

        1. Compute loss

        2. Compute gradient multipliers (if transfer learning from classification task) and multiply gradients with gradient multipliers

        3. Update Gradient descent

    * Example code:
    
    ```
    with tf.device(config.variables_device()):
        total_loss, grads_and_vars = model_deploy.optimize_clones(clones, optimizer)
        total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Modify the gradients for biases and last layer variables.
        last_layers = model.get_extra_layer_scopes(FLAGS.last_layers_contain_logits_only)
        grad_mult = train_utils.get_model_gradient_multipliers(last_layers, FLAGS.last_layer_gradient_multiplier)
        if grad_mult:
            grads_and_vars = slim.learning.multiply_gradients(grads_and_vars, grad_mult)

        # Create gradient update op.
        grad_updates = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')
    ```

**Test**:

* [ ] DeeplabV3Augmentator

## 18/03/2020

**Notes**: https://github.com/CSAILVision/semantic-segmentation-pytorch

* Optimizer: uses different optimizers for encoder and decoder

    * Both optimizers are SGD
    
    * Only apply weight decay for weights of nn.Linear nn.Conv2d, their biases terms has no weight decay, and BN layer has no weight decay
    
    * Base LR is 0.02
    
    * Example code:
    
    ```
    def group_weight(module):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                if m.weight is not None:
                    group_no_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)

        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
        return groups


    def create_optimizers(nets, cfg):
        (net_encoder, net_decoder, crit) = nets
        optimizer_encoder = torch.optim.SGD(
            group_weight(net_encoder),
            lr=cfg.TRAIN.lr_encoder,
            momentum=cfg.TRAIN.beta1,
            weight_decay=cfg.TRAIN.weight_decay)
        optimizer_decoder = torch.optim.SGD(
            group_weight(net_decoder),
            lr=cfg.TRAIN.lr_decoder,
            momentum=cfg.TRAIN.beta1,
            weight_decay=cfg.TRAIN.weight_decay)
        return (optimizer_encoder, optimizer_decoder)
    ```
    
* Init weights
    
    * Init method:
    
    ```
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    ```

* Model
    
    * Use Deep supervision (i.e. auxiliary loss)
    
    ```
    if self.deep_sup_scale is not None: # use deep supervision technique
        (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
    else:
        pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))

    loss = self.crit(pred, feed_dict['seg_label'])
    if self.deep_sup_scale is not None:
        loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
        loss = loss + loss_deepsup * self.deep_sup_scale
    ```
    
    * Loss function: only NLLLoss from PyTorch
    * Metric: Pixel accuracy (for training)
    
    ```
    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc
    ```

* Metric during evaluation

```
def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)
```

**Note**: https://github.com/mrgloom/awesome-semantic-segmentation

# Lecture outline

## Coding lecture schedule

There are totally 9 coding sessions. Each session lasts for 2 - 3 hours.

* Session 1: Explore VOC-2012 dataset for segmantic segmentation

* Session 2: Write torch Dataset for VOC-2012 dataset

* Session 3: Write trainer class for training segmentation model on VOC-2012 dataset

* Session 4: Write model class for training segmentation model on VOC-2012 dataset

* Session 5: Write optimizers, learning rate scheduler, loss functions, and metrics for training and evaluation

* Session 6: Build an end-to-end training pipeline for semantic segmentation

* Session 7: Write evaluation script and visualization code for VOC-2012 dataset

* Session 8: Play around with models, loss functions, metrics, optimizers, and schedulers (2)

* Session 9: Hold a small inner-class competition based on VOC-2012 dataset

## Lectures

* Session 1: Introduction to Semantic Segmentation

    * Theory: Introduction to  semantic segmentation and VOC-2012 dataset
    
    * Coding: Explore VOC-2012 dataset with Python

* Session 2: Implement torch Dataset for VOC-2012 dataset

* Session 3: Popular segmentation models
    
    * Theory: 
        
        * UNet and its relatives (e.g. FCN, LinkNet, etc.)
        
        * DeeplabV3 and its relatives (e.g. PSPNet, etc.)
    
    * Coding: Implement UNet with PyTorch

* Topic 4: Loss functions for semantic segmentation

    * Theory: 
        
        * Cross-Entropy loss, Weighted Cross-Entropy loss, Focal loss
        
        * DICE loss, HD loss, soft IoU loss
    
    * Coding: Implement loss functions in PyTorch

* Topic 5: Building a Trainer for semantic segmentation

    * Theory: 
        
        * Commonly used optimizers and schedulers in semantic segmentation
        
        * Commonly used training pipeline for semantic segmentation model
    
    * Coding: 
        
        * Implement optimizers and schedulers for semantic segmentation
        
        * Build a simple training pipeline for semantic segmentation, i.e. a `Trainer` class including
        
            * `__get_model`, `__get_optimizer`, `__get_dataloaders` methods
        
            * `train_one_epoch` method
            
            * `eval_one_epoch` method

* Topic 6: Class imbalance problem in VOC-2012 segmentation datasset, and metrics to evaluate semantic segmentation model
    
    * Theory:
        
        * Symptoms of class imbalance - high IoU scores and low loss values
        
        * IoU, Precision, and Recall
    
    * Coding:
        
        * Implement metric functions to evaluate semantic segmentation model
        
        * Integrate metric functions into the trainer class
        
        * Refine the trainer class for better visualization and best-model saving

* Topic 7: Visualization and debugging
    
    * Theory:
        
        * Review about semantic segmentation and training-and-evaluation pipeline
    
    * Coding:
        
        * Visualize model predictions with Python
        
        * Debug semantic segmentation model

* Topic 8: Play around with semantic segmentation
    
    * Theory:
        
        * Brain-storm session about semantic segmentation modeling
        
        * Brain-storm session about loss functions for semantic segmentation
    
    * Coding:
        
        * Wrap trainer class to flexibly try out different models
        
        * Try different semantic segmentation models
        
        * Try different loss functions for semantic segmentation

* Topic 9: Hold a class-level competition about semantic segmentation with VOC-2012 dataset
    
    * Rules:
        
        * Students brain-storm or even implement their ideas, e.g. models, loss functions, etc. beforehand at home
        
        * Students continue working on the competition at the class
        
        * The last 1 - 2 hours, depending on the number of students / teams, will be dedicated for evaluation and solution presentation