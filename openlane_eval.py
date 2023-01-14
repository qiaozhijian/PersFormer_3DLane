import os

openlane_dir = '/media/qzj/Document/datasets/OpenLane'
eval_2d_bin = '/home/qzj/code/OpenLane/eval/LANE_evaluation/lane2d/evaluate'
eval_3d_py = '/home/qzj/code/OpenLane/eval/LANE_evaluation/lane3d/eval_3D_lane.py'

def eval_openlane_cases():
    cases = ['up_down_case','curve_case','extreme_weather_case','intersection_case','merge_split_case','night_case']
    for case in cases:
        result_dir = os.path.abspath(os.path.join('data_splits', 'openlane', 'PersFormer_{}'.format(case)))
        result_2d_dir = os.path.join(result_dir, 'result_2d', 'validation')
        result_3d_dir = os.path.join(result_dir, 'result_3d', 'validation')

        # generate test list txt for both 2d and 3d
        test_items = []
        list_2d = os.listdir(os.path.join(result_2d_dir))
        for i in range(len(list_2d)):
            jsons_2d = os.listdir(os.path.join(result_2d_dir, list_2d[i]))
            for j in range(len(jsons_2d)):
                test_items.append(os.path.join(list_2d[i], jsons_2d[j]))
        test_list = os.path.join(result_dir, 'test_list.txt')
        with open(test_list, 'w') as f:
            for i, item in enumerate(test_items):
                item = item.replace('json', 'jpg')
                end_str = '\n' if i < len(test_items) - 1 else ''
                f.write(item + end_str)
            f.close()

        # 2d evaluation
        annotation_dir = os.path.join(openlane_dir, 'lane3d_1000/test', case) + '/'
        result_2d_dir = result_2d_dir + '/'
        image_dir = os.path.join(openlane_dir, 'images', 'validation') + '/'
        output_dir = os.path.join(result_dir, 'eval_2d') + '/'
        eval_2d_cmd(eval_2d_bin, annotation_dir, result_2d_dir, image_dir, test_list, output_dir)

        # 3d evaluation
        result_3d_dir = result_3d_dir + '/'
        output_dir = os.path.join(result_dir, 'eval_3d') + '/'
        eval_3d_cmd(eval_3d_py, annotation_dir, result_3d_dir, test_list, output_dir)

        
def eval_2d_cmd(eval_2d_bin, annotation_dir, result_2d_dir, image_dir, test_list, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cmd = "{} -a {} -d {} -i {} -l {} -o {}".format(
        eval_2d_bin, annotation_dir, result_2d_dir, image_dir, test_list, output_dir)
    print(cmd)
    os.system(cmd)

def eval_3d_cmd(eval_3d_py, annotation_dir, result_3d_dir, test_list, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cmd = "python {} --dataset_dir={} --pred_dir={} --test_list={} --output_dir={}".format(
        eval_3d_py, annotation_dir, result_3d_dir, test_list, output_dir)
    print(cmd)
    os.system(cmd)

if __name__ == '__main__':

    eval_openlane_cases()