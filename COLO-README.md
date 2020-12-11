# Important information

* Description info (object counts and classes) for visual genome scene graphs: `datasets/vg_bm/VG-SGG-dicts.jsondatasets/vg_bm/VG-SGG-dicts.json`

# Results files `results/*`

* `predictions.pth`
    * List of BoxList, see `lib/scene_parser/rcnn/structures/bounding_box_pair.py`
    ```
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    ```
    * extra field on each BoxList object:
      * ['labels', 'scores', 'logits', 'features']
      * 'labels' object labels. the numbers here correspond labels in `results/test_description.pth` (Colorado made this change)
      * 'scores' score for object label
      * 'logits': the logit scores for each output
      * 'features': pooled feature vector for the individual object

* `predictions_pred.pth`: an array that contains predicate predictions for each image
    * List of BoxPairList, see `lib/scene_parser/rcnn/structures/bounding_box_pair.py`
    * extra field on each box pair object:
      * ['idx_pairs', 'scores']
      * 'idx_pairs' is the pairs of objects, where each object id is in the associated `BoxList.labels` field from `prediction.pth` (be careful here, a 0 on one `BoxPairList` could be an airplane and a 0 on another `BoxPairList` could be a dog. You have to check the associated `BoxList`
      * 'scores' is the scores for the relationships, where it will take dimensions `nrows(idx_pairs) x number of relationships`. The relationship ids are on the testing dataset object (and due to changes Colorado made to the codebase, they are also output into `results/test_description.pth`
      
* coco_results.pth: a tiny pth file containing the AP (average precision) results
  * coco_results.results['bbox']
            * results: {AP, AP50, AP75, APs, APm, APl}
* bbox.json: (subset of info in `predictions.pth`) contains a json of all bounding boxes for each image, e.g. for image 0, here is the classification of 135 with score 0.98: 
``` json
  {
	'image_id': 0,
  	'category_id': 135,
  	'bbox': [274.5951232910156, 63.055023193359375, 681.42626953125, 678.501220703125],
	'score': 0.97893720
  }
```

* `test_description.pth` (colorado made this file): descripes the test dataset. An object with a dictionary called `description` with the following fields:
    * class_to_ind: object classes to index
    * ind_to_classes: indices to object classes
    * ind_to_predicates: indices to predicates
    * predicate_to_ind: predicates to indices
    * image_file: the image file used
    * roidb_file: the roidb file used
    * im_sizes: the size of the testing images
    * date: the date the test results were saved

* `objectlabels`: the ordered list of object labels corresponding to the label numbers in `predictions.pth`
* `predicates`: the ordered list of predicates (relationships) corresponding to the numbers in `predictions_pred.pth.get_field('idx_pairs')` (only present when config.MODEL.RELATION_ON is true)

# Detailed Data Description:

* `VG-SGG.h5`
  * ['active_object_mask', 'boxes_1024', 'boxes_512', 'img_to_first_box', 'img_to_first_rel', 'img_to_last_box', 'img_to_last_rel', 'labels', 'predicates', 'relationships', 'split']
* `imdb_1024.h5`
  * ['active_object_mask', 'boxes_1024', 'boxes_512', 'img_to_first_box', 'img_to_first_rel', 'img_to_last_box', 'img_to_last_rel', 'labels', 'predicates', 'relationships', 'split']
* `proposals.h5`
  * ['im_scales', 'im_to_roi_idx', 'num_rois', 'rpn_rois', 'rpn_scores']

# Testing on new image datasets

## What you need to get started
We're going to be shimming the data processing into the instructions here: https://github.com/danfeiX/scene-graph-TF-release/tree/master/data_tools

* A directory containing jpgs
* A "meta json" file, containing an array of json objects, where each json object has at least the following key-value:
  { "image_id": "the-image-id" }
  where the image is in a file called `the-image-id.jpg`
  OR
  provide { "file_name": "some-file-name.jpg"} (but you have to be using colorado's variant of the data tools in scene_graph_TF_release)
  
  A lot of datasets come with some version of a meta file, so for example, you can use `jq` to easily add/transform to get this field. For COCO, I did this:
  ```
  jq '[.images | .[] | .["image_id"] = .id]' < coco/annotations/instances_val2017.json > coco-val-meta.json
  ```


## Prep the data
Currently an example with the coco dataset:
1. python vg_to_imdb.py --image_dir 'coco/val2017' --imh5_dir 'cocoout' --num_workers 40 --metadata_input 'coco-val-meta.json'

# Misc notes

* MSDN: Multi-level Scene Description Network is an alternative scene graph generation model, see here: https://github.com/yikang-li/MSDN
* FactorizableNet has comparable code in terms of library state: https://github.com/yikang-li/FactorizableNet (this is the evolution of MSDN)

