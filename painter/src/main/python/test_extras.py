
import os
import shutil
from PyQt6 import QtCore


sync_dir = os.path.join(os.getcwd(), 'test_rp_sync')


def test_specify_seg(qtbot):
    """ test we can click the specify_seg_btn without an error """
    from mask_images import MaskImWidget
    # initialise the mask im widget
    mask_im_widget = MaskImWidget()
    mask_im_widget.show()
    qtbot.mouseClick(mask_im_widget.specify_seg_btn, QtCore.Qt.MouseButton.LeftButton)



def dl_dir_from_zip(url, output_path):
    import urllib.request
    import zipfile
    import glob
    # if the directory does not exist, assume it needs downloading
    if not os.path.isdir(output_path):
        print('downloading', url)
        urllib.request.urlretrieve(url, 'temp.zip')
        with zipfile.ZipFile("temp.zip", "r") as zip_ref:
            zip_ref.extractall(output_path)

        # remove the junk osx metadata that was in the zip file
        junk_osx_dir = os.path.join(output_path, '__MACOSX')

        if os.path.isdir(junk_osx_dir):
            shutil.rmtree(junk_osx_dir)

        os.remove(os.path.join(os.getcwd(), 'temp.zip'))
        
        # remove the parent folder - we just want the files in the target folder.
        parent_folder = url.split('/')[-1].replace('.zip', '')
        all_files = glob.glob(os.path.join(output_path, parent_folder, '*.*'), recursive=True)
        for file_path in all_files:
            dst_path = os.path.join(output_path, os.path.basename(file_path))
            shutil.move(file_path, dst_path)
        shutil.rmtree(os.path.join(output_path, parent_folder))



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

    # prepare segmentations
    results_dir = os.path.join(sync_dir, 'projects', 'biopores_corrective_a', 'results')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    biopore_seg_url = 'https://zenodo.org/record/8037046/files/user_a_corrective_biopores_750_training_seg_model_33.zip'
    seg_dir = os.path.join(results_dir, 'seg_model_33')
    dl_dir_from_zip(biopore_seg_url, seg_dir)


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

    qtbot.waitUntil(check_output, timeout=20000)


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

    qtbot.waitUntil(check_output, timeout=20000)


