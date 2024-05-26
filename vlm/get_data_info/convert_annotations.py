annotation{
"id": int, "image_id": int, "category_id": int, "segmentation": RLE or [polygon], "area": float, "bbox": [x,y,width,height], "iscrowd": 0 or 1,
}

categories[{
"id": int, "name": str, "supercategory": str,
}]

second = {"info": [], "licenses": [], "images": [{"file_name": "2354786.jpg", "height": 270, "width": 500, "id": 486488, "original_id": "2354786", "caption": "two cars on street. the name of the building. a traffic light. Pillars on the building. the doors are shut. red lights engaged by using breaks. the monument is green in colour. stop light has a red light displayed. outside of the USA license plate. a parking sign.", "tokens_negative": [[0, 3], [4, 8], [9, 11], [12, 18], [20, 23], [24, 28], [29, 31], [36, 44], [46, 47], [48, 55], [56, 61], [63, 70], [71, 73], [88, 91], [92, 97], [98, 101], [102, 106], [108, 111], [112, 118], [119, 126], [127, 129], [130, 135], [144, 147], [148, 156], [157, 159], [160, 165], [166, 168], [177, 181], [188, 191], [192, 193], [194, 197], [198, 203], [204, 213], [215, 247], [249, 250], [251, 258], [259, 263]], "data_source": "vg", "dataset_name": "mixed"}
          
                                                 
{"file_name": "4886839805.jpg", "height": "500", "width": "371", "id": 27, "caption": "2 wet boys in blue swimming suits", "dataset_name": "flickr", "tokens_negative": [[0, 33]], "sentence_id": 2, "original_img_id": 4886839805, "tokens_positive_eval": [[[0, 10]], [[14, 33]]]}
                                                 
# converting to second format is readable by yolo
# find stopwords, space and etc index range and append to the tokens negative
# fill info other than data source and dataset name