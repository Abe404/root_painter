
import os
import shutil
from PyQt6 import QtCore

# sync directory for use with tests
sync_dir = os.path.join(os.getcwd(), 'test_rp_sync')
timeout_ms = 20000


def dl_dir_from_zip(url, output_path):
    """ download a zip from url and place contents in output_path """
    import urllib.request
    import zipfile
    import glob
    # if the directory does not exist, assume it needs downloading
    if not os.path.isdir(output_path):
        print('downloading', url)
        urllib.request.urlretrieve(url, 'temp.zip')
        with zipfile.ZipFile("temp.zip", "r") as zip_ref:
            zip_ref.extractall('temp_zip_output')

        # remove the junk osx metadata that was in the zip file
        junk_osx_dir = os.path.join('temp_zip_output', '__MACOSX')

        if os.path.isdir(junk_osx_dir):
            shutil.rmtree(junk_osx_dir)

        os.remove(os.path.join(os.getcwd(), 'temp.zip'))
        
        zip_dir = os.listdir(os.path.join('temp_zip_output'))[0]
        zip_path = os.path.join('temp_zip_output', zip_dir)
        shutil.move(zip_path, output_path)
        shutil.rmtree('temp_zip_output')


def setup_function():
    import urllib.request
    import zipfile
    import shutil
    print('running setup')
    # prepare biopores training dataset
    datasets_dir = os.path.join(sync_dir, 'datasets')
    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)
    biopore_url = 'https://zenodo.org/record/3754046/files/biopores_750_training.zip'
    bp_dataset_dir = os.path.join(datasets_dir, 'biopores_750_training')
    dl_dir_from_zip(biopore_url, bp_dataset_dir)

    results_dir = os.path.join(sync_dir, 'projects', 'biopores_corrective_a', 'results')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # prepare segmentations
    biopore_seg_url = 'https://zenodo.org/record/8037046/files/user_a_corrective_biopores_750_training_seg_model_33.zip'
    seg_dir = os.path.join(results_dir, 'seg_model_33')
    dl_dir_from_zip(biopore_seg_url, seg_dir)


    # prepare annotations
    biopore_annot_url = 'https://zenodo.org/record/8041842/files/user_a_corrective_biopores_750_training_annotation.zip'
    annot_dir = os.path.join(sync_dir, 'projects', 'biopores_corrective_a', 'annotations')
    dl_dir_from_zip(biopore_annot_url, annot_dir)


def test_specify_seg_for_mask_widget(qtbot):
    """ test we can click the specify_seg_btn without an error """
    from mask_images import MaskImWidget
    # initialise the mask im widget
    mask_im_widget = MaskImWidget()
    mask_im_widget.show()
    qtbot.mouseClick(mask_im_widget.specify_seg_btn, QtCore.Qt.MouseButton.LeftButton)


def test_mask_operation(qtbot):
    from mask_images import MaskImWidget
    mask_widget = MaskImWidget()
    mask_widget.show()
    results_dir = os.path.join(sync_dir, 'projects', 'biopores_corrective_a', 'results')
    seg_dir = os.path.join(results_dir, 'seg_model_33')
    masked_dir = os.path.join(results_dir, 'masked_model_33')

    # if the masked_dir already exists then delete it - 
    # we want to test creating it and making the output.
    if os.path.isdir(masked_dir):
        shutil.rmtree(masked_dir)

    dataset_dir = os.path.join(sync_dir, 'datasets', 'biopores_750_training')
    mask_widget.seg_dir = seg_dir
    mask_widget.im_dir = dataset_dir
    mask_widget.out_dir = masked_dir
    mask_widget.validate()
    mask_widget.submit_btn.click()

    def check_output():
        if not os.path.isdir(mask_widget.out_dir):
            return False
        return len(os.listdir(mask_widget.out_dir)) == len(os.listdir(mask_widget.seg_dir))

    qtbot.waitUntil(check_output, timeout=timeout_ms)


def test_specify_seg_comp(qtbot):
    """ test we can click the specify_seg_btn without an error """
    from extract_comp import ExtractCompWidget
    # initialise the mask im widget
    widget = ExtractCompWidget()
    widget.show()
    qtbot.mouseClick(widget.specify_seg_btn, QtCore.Qt.MouseButton.LeftButton)


def test_extract_composites(qtbot):
    from extract_comp import ExtractCompWidget

    extract_comp_widget = ExtractCompWidget()
    extract_comp_widget.show()
    dataset_dir = os.path.join(sync_dir, 'datasets', 'biopores_750_training')
    results_dir = os.path.join(sync_dir, 'projects', 'biopores_corrective_a', 'results')
    seg_dir = os.path.join(results_dir, 'seg_model_33')
    comp_dir = os.path.join(results_dir, 'comp_model_33')

    # If the dir already exists then delete it.
    # We want to test creating it and making the output.
    if os.path.isdir(comp_dir):
        shutil.rmtree(comp_dir)

    extract_comp_widget.seg_dir = seg_dir
    extract_comp_widget.im_dir = dataset_dir
    extract_comp_widget.comp_dir = comp_dir
    extract_comp_widget.validate()
    extract_comp_widget.submit_btn.click()

    def check_output():
        if not os.path.isdir(extract_comp_widget.comp_dir):
            return False
        return (len(os.listdir(extract_comp_widget.comp_dir)) == 
                len(os.listdir(extract_comp_widget.seg_dir)))

    qtbot.waitUntil(check_output, timeout=timeout_ms)


def test_specify_seg_btn_for_rve_widget(qtbot):
    """ test we can click the specify_seg_btn without an error """
    from convert_seg import ConvertSegWidget
    from convert_seg import convert_seg_to_rve
    # initialise the mask im widget
    widget = ConvertSegWidget(convert_seg_to_rve,
                              'RhizoVision Explorer compatible format')
    widget.show()
    qtbot.mouseClick(widget.specify_seg_btn, QtCore.Qt.MouseButton.LeftButton)


def test_extract_rve(qtbot):
    from convert_seg import ConvertSegWidget
    from convert_seg import convert_seg_to_rve
    widget = ConvertSegWidget(convert_seg_to_rve,
                              'RhizoVision Explorer compatible format')
    widget.show()

    results_dir = os.path.join(sync_dir, 'projects', 'biopores_corrective_a', 'results')
    seg_dir = os.path.join(results_dir, 'seg_model_33')
    out_dir = os.path.join(results_dir, 'rve_model_33')

    # If the dir already exists then delete it.
    # We want to test creating it and making the output.
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    widget.seg_dir = seg_dir
    widget.out_dir = out_dir
    widget.validate()
    widget.submit_btn.click()

    def check_output():
        if not os.path.isdir(widget.out_dir):
            return False
        return (len(os.listdir(widget.out_dir)) == 
                len(os.listdir(widget.seg_dir)))

    qtbot.waitUntil(check_output, timeout=timeout_ms)


def test_convert_seg_to_annotations(qtbot):
    from convert_seg import ConvertSegWidget
    from convert_seg import convert_seg_to_annot
    widget = ConvertSegWidget(convert_seg_to_annot, 'annotations')
    widget.show()

    results_dir = os.path.join(sync_dir, 'projects', 'biopores_corrective_a', 'results')
    seg_dir = os.path.join(results_dir, 'seg_model_33')
    out_dir = os.path.join(results_dir, 'annot_from_seg_model_33')

    # If the dir already exists then delete it.
    # We want to test creating it and making the output.
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    widget.seg_dir = seg_dir
    widget.out_dir = out_dir
    widget.validate()
    widget.submit_btn.click()

    def check_output():
        if not os.path.isdir(widget.out_dir):
            return False
        return (len(os.listdir(widget.out_dir)) == 
                len(os.listdir(widget.seg_dir)))

    qtbot.waitUntil(check_output, timeout=timeout_ms)


def test_assign_corrections(qtbot):
    from assign_corrections import AssignCorrectionsWidget
    widget = AssignCorrectionsWidget()
    widget.show()

    results_dir = os.path.join(sync_dir, 'projects', 'biopores_corrective_a', 'results')
    annot_dir = os.path.join(sync_dir, 'projects', 'biopores_corrective_a', 'annotations')
    seg_dir = os.path.join(results_dir, 'seg_model_33')
    out_dir = os.path.join(results_dir, 'corrected_seg')

    # If the dir already exists then delete it.
    # We want to test creating it and making the output.
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    widget.annot_dir = annot_dir
    widget.seg_dir = seg_dir
    widget.out_dir = out_dir
    widget.validate()
    widget.submit_btn.click()

    def check_output():
        if not os.path.isdir(widget.out_dir):
            return False

        out_files = os.listdir(widget.out_dir)
        in_files = os.listdir(widget.annot_dir)
        in_files = [i for i in in_files if os.path.splitext(i)[1] == '.png']

        return (len(out_files) == 
                len(in_files))

    qtbot.waitUntil(check_output, timeout=timeout_ms)


def test_create_random_split(qtbot):
    from random_split import RandomSplitWidget
    widget = RandomSplitWidget()
    widget.show()
    dataset_dir = os.path.join(sync_dir, 'datasets', 'biopores_750_training')
    out_dir = os.path.join(sync_dir, 'datasets', 'bp_split')

    # If the dir already exists then delete it.
    # We want to test creating it and making the output.
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    widget.source_dir = dataset_dir
    widget.output_dir = out_dir
    widget.validate()
    widget.create_btn.click()

    def check_output():
        if not os.path.isdir(widget.output_dir):

            return False

        if not os.path.isdir(os.path.join(widget.output_dir, 'split_1')):
            return False

        if not os.path.isdir(os.path.join(widget.output_dir, 'split_2')):
            return False 

        out_files = os.listdir(os.path.join(widget.output_dir, 'split_1'))
        out_files += os.listdir(os.path.join(widget.output_dir, 'split_2'))
        in_files = os.listdir(widget.source_dir)
        return (len(out_files) == 
                len(in_files))

    qtbot.waitUntil(check_output, timeout=timeout_ms)

 
def test_resize_images(qtbot):
    from resize_images import ResizeWidget
    widget = ResizeWidget()
    widget.show()
    dataset_dir = os.path.join(sync_dir, 'datasets', 'biopores_750_training')
    out_dir = os.path.join(sync_dir, 'datasets', 'bp_resized')

    # If the dir already exists then delete it.
    # We want to test creating it and making the output.
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    widget.source_dir = dataset_dir
    widget.output_dir = out_dir
    widget.validate()
    widget.create_btn.click()

    def check_output():
        if not os.path.isdir(widget.output_dir):
            return False
        out_files = os.listdir(widget.output_dir)
        in_files = os.listdir(widget.source_dir)
        return (len(out_files) == 
                len(in_files))

    qtbot.waitUntil(check_output, timeout=timeout_ms)
