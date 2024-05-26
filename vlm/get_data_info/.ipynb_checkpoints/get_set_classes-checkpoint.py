#iterate through the vlm.jsonl
# get classes
# Add to coco classes and edit the finetune script
file = '../../novice/vlm.jsonl'
import json

def main():
    with open(file, 'r') as fp:
        result = [json.loads(jline) for jline in fp.read().splitlines()]
    class_to_use = []
    for l in result:
        class_to_use.extend([x['caption'] for x in l['annotations']])
    #print(len(set(class_to_use)), sum(len(l) for l in result))
    #print(set(class_to_use))
    class_to_use = list(set(class_to_use))
    id2label = {index: x for index, x in enumerate(class_to_use, start=0)}
    label2id = {v: k for k, v in id2label.items()}
    instance_id = 0
    for i, r in enumerate(result, start = 0):
        if len(r['annotations']) > 1:
            is_crowd = 1
        else:
            is_crowd = 0
        #add image id
        r['image_id'] = i
        #add category id
        for annot in r['annotations']:
            #bbox id
            annot['category_id'] = label2id[annot['caption']]
            #is_crowd
            annot['is_crowd'] = is_crowd
            #area
            annot['area'] = annot['bbox'][-1] * annot['bbox'][-2]
    #print(result)
    json_dump = {}
    json_dump['info'] = []
    json_dump['licenses'] = []
    json_dump['images'] = result
    with open('new_annot.json', 'w') as fp:
        json.dump(json_dump, fp)
    with open('new_label.json', 'w') as fp:
        json.dump(id2label, fp)
    
    #create a class text file
    
if __name__ == '__main__':
    main()