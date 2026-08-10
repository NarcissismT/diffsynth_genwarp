from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('DiffSynth-Studio/Qwen-Image-Self-Generated-Dataset', subset_name='default', split='train')

print(ds)