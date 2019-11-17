import glob
import zipfile
import os, errno
from random import shuffle
from shutil import copy2
import argparse
import shutil

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

def _main_(args):
	import sys
	print(sys.argv[1:])
	
	data_dir='Data\\'
	annot_dir='lib\\annotation\\'
	annot_dir='lib\\annotation\\'
	tmp_dir='tmp'
	#main_data='Data\data.zip'
	main_data=args[0]
	dir1='Data\Train\Best_hyperparameter_80_percent\\'
	dir2='Data\Validation\Validation_10_percent\\'
	dir3='Data\Test\Test_10_percent\\'
	dir4='Data\Train\\Under_10_min_training\\'
	dir5='Data\Train\\Under_90_min_tuning\\'
	dir6='Data\Validation\\3_samples\\'
	#dir3='Data\Test\Test_10_percent\\'
	#dir4='Data\Train\\Under_10_min_training\\'
	#dir5='Data\Train\Under_90_min_tuning\\'
	#print(dir4)
	
	try:
		os.makedirs(os.path.join('.',tmp_dir))
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise
	
	dz=zipfile.ZipFile(main_data,'r')
	dz.extractall()

	all_img_paths_yolo=glob.glob(os.path.join(data_dir,'Training_yolo\*'))
	all_img_paths_cnn=glob.glob(os.path.join(data_dir,'Training_cnn\*\*'))

	#print(all_img_paths_yolo[0])
	shuffle(all_img_paths_yolo)
	shuffle(all_img_paths_cnn)
	#print(all_img_paths_cnn[0])


	all_annot_paths_yolo={}
	for path in all_img_paths_yolo:
		all_annot_paths_yolo[path]=os.path.join(annot_dir,path.split('\\')[-1][:-3])+'xml'

	dirs=[dir1, dir2, dir3, dir4, dir5, dir6]

	for dir in dirs:
		try:
			os.makedirs(os.path.join(dir[:-1],'train_yolo'))
			os.makedirs(os.path.join(dir[:-1],'annotation'))
			os.makedirs(os.path.join(dir[:-1],'train_cnn\\00000'))
			os.makedirs(os.path.join(dir[:-1],'train_cnn\\00001'))
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise
				
	#train separation
	for i in range(0,int(len(all_img_paths_yolo)*0.8)):
		path=all_img_paths_yolo[i]
		#print(path)
		copydir=os.path.join(dir1,'train_yolo',path.split('\\')[-1])
		#copydir=copydir.replace('\\','/')
		#copydir=copydir.replace('/','\\')
		#print(copydir)
		copy2(path,copydir)
		annotpath=all_annot_paths_yolo[path]
		copyannotpath=os.path.join(dir1,'annotation',annotpath.split('\\')[-1])
		copy2(annotpath,copyannotpath)
	#print(len(copydir))

	#validation separation
	for i in range(int(len(all_img_paths_yolo)*0.8),int(len(all_img_paths_yolo)*0.9)):
		path=all_img_paths_yolo[i]
		#print(path)
		copydir=os.path.join(dir2,'train_yolo',path.split('\\')[-1])
		#copydir=copydir.replace('\\','/')
		#copydir=copydir.replace('/','\\')
		#print(copydir)
		copy2(path,copydir)
		annotpath=all_annot_paths_yolo[path]
		copyannotpath=os.path.join(dir2,'annotation',annotpath.split('\\')[-1])
		copy2(annotpath,copyannotpath)
		
	#test separation
	for i in range(int(len(all_img_paths_yolo)*0.9),int(len(all_img_paths_yolo))):
		path=all_img_paths_yolo[i]
		#print(path)
		copydir=os.path.join(dir3,'train_yolo',path.split('\\')[-1])
		#copydir=copydir.replace('\\','/')
		#copydir=copydir.replace('/','\\')
		#print(copydir)
		copy2(path,copydir)
		annotpath=all_annot_paths_yolo[path]
		copyannotpath=os.path.join(dir3,'annotation',annotpath.split('\\')[-1])
		copy2(annotpath,copyannotpath)

	shuffle(all_img_paths_yolo)
	all_img_paths_yolo2=all_img_paths_yolo[:150]
	#under 10 min
	for i in range(0,int(len(all_img_paths_yolo2))):
		path=all_img_paths_yolo2[i]
		#print(path)
		copydir=os.path.join(dir4,'train_yolo',path.split('\\')[-1])
		#copydir=copydir.replace('\\','/')
		#copydir=copydir.replace('/','\\')
		#print(copydir)
		copy2(path,copydir)
		annotpath=all_annot_paths_yolo[path]
		copyannotpath=os.path.join(dir4,'annotation',annotpath.split('\\')[-1])
		copy2(annotpath,copyannotpath)
		
	shuffle(all_img_paths_yolo)
	all_img_paths_yolo2=all_img_paths_yolo[:250]
	#under 90 min
	for i in range(0,int(len(all_img_paths_yolo2))):
		path=all_img_paths_yolo2[i]
		#print(path)
		copydir=os.path.join(dir5,'train_yolo',path.split('\\')[-1])
		#copydir=copydir.replace('\\','/')
		#copydir=copydir.replace('/','\\')
		#print(copydir)
		copy2(path,copydir)
		annotpath=all_annot_paths_yolo[path]
		copyannotpath=os.path.join(dir5,'annotation',annotpath.split('\\')[-1])
		copy2(annotpath,copyannotpath)
		
	shuffle(all_img_paths_yolo)
	all_img_paths_yolo2=all_img_paths_yolo[:3]
	#3 val
	for i in range(0,int(len(all_img_paths_yolo2))):
		path=all_img_paths_yolo2[i]
		#print(path)
		copydir=os.path.join(dir6,'train_yolo',path.split('\\')[-1])
		#copydir=copydir.replace('\\','/')
		#copydir=copydir.replace('/','\\')
		#print(copydir)
		copy2(path,copydir)
		annotpath=all_annot_paths_yolo[path]
		copyannotpath=os.path.join(dir6,'annotation',annotpath.split('\\')[-1])
		copy2(annotpath,copyannotpath)
		
	for i in range(0,int(len(all_img_paths_cnn)*0.8)):
		path=all_img_paths_cnn[i]
		#print(path)
		copydir=os.path.join(dir1,'train_cnn',path.split('\\')[-2],path.split('\\')[-1])
		#copydir=copydir.replace('\\','/')
		#copydir=copydir.replace('/','\\')
		#print(copydir)
		copy2(path,copydir)
		
	for i in range(int(len(all_img_paths_cnn)*0.8),int(len(all_img_paths_cnn)*0.9)):
		path=all_img_paths_cnn[i]
		#print(path)
		copydir=os.path.join(dir1,'train_cnn',path.split('\\')[-2],path.split('\\')[-1])
		#copydir=copydir.replace('\\','/')
		#copydir=copydir.replace('/','\\')
		#print(copydir)
		copy2(path,copydir)
		
	for i in range(int(len(all_img_paths_cnn)*0.9),int(len(all_img_paths_cnn))):
		path=all_img_paths_cnn[i]
		#print(path)
		copydir=os.path.join(dir1,'train_cnn',path.split('\\')[-2],path.split('\\')[-1])
		#copydir=copydir.replace('\\','/')
		#copydir=copydir.replace('/','\\')
		#print(copydir)
		copy2(path,copydir)
		
		
	for i in range(0,int(len(all_img_paths_cnn)*0.8)):
		path=all_img_paths_cnn[i]
		#print(path)
		copydir=os.path.join(dir4,'train_cnn',path.split('\\')[-2],path.split('\\')[-1])
		#copydir=copydir.replace('\\','/')
		#copydir=copydir.replace('/','\\')
		#print(copydir)
		copy2(path,copydir)
		
	for i in range(int(len(all_img_paths_cnn)*0.8),int(len(all_img_paths_cnn)*0.9)):
		path=all_img_paths_cnn[i]
		#print(path)
		copydir=os.path.join(dir4,'train_cnn',path.split('\\')[-2],path.split('\\')[-1])
		#copydir=copydir.replace('\\','/')
		#copydir=copydir.replace('/','\\')
		#print(copydir)
		copy2(path,copydir)
		
	for i in range(int(len(all_img_paths_cnn)*0.9),int(len(all_img_paths_cnn))):
		path=all_img_paths_cnn[i]
		#print(path)
		copydir=os.path.join(dir4,'train_cnn',path.split('\\')[-2],path.split('\\')[-1])
		#copydir=copydir.replace('\\','/')
		#copydir=copydir.replace('/','\\')
		#print(copydir)
		copy2(path,copydir)
		
	for i in range(0,int(len(all_img_paths_cnn)*0.8)):
		path=all_img_paths_cnn[i]
		#print(path)
		copydir=os.path.join(dir5,'train_cnn',path.split('\\')[-2],path.split('\\')[-1])
		#copydir=copydir.replace('\\','/')
		#copydir=copydir.replace('/','\\')
		#print(copydir)
		copy2(path,copydir)
		
	for i in range(int(len(all_img_paths_cnn)*0.8),int(len(all_img_paths_cnn)*0.9)):
		path=all_img_paths_cnn[i]
		#print(path)
		copydir=os.path.join(dir5,'train_cnn',path.split('\\')[-2],path.split('\\')[-1])
		#copydir=copydir.replace('\\','/')
		#copydir=copydir.replace('/','\\')
		#print(copydir)
		copy2(path,copydir)
		
	for i in range(int(len(all_img_paths_cnn)*0.9),int(len(all_img_paths_cnn))):
		path=all_img_paths_cnn[i]
		#print(path)
		copydir=os.path.join(dir5,'train_cnn',path.split('\\')[-2],path.split('\\')[-1])
		#copydir=copydir.replace('\\','/')
		#copydir=copydir.replace('/','\\')
		#print(copydir)
		copy2(path,copydir)
	
	
	for dir in dirs:
		curzip='data'
		shutil.make_archive('data','zip',dir)
		copy2('data.zip',os.path.join(dir,'data.zip'))
		os.remove('data.zip')
		shutil.rmtree(os.path.join(dir,'annotation'))
		shutil.rmtree(os.path.join(dir,'train_yolo'))
		shutil.rmtree(os.path.join(dir,'train_cnn'))
		
	shutil.rmtree(os.path.join(data_dir,'Training_cnn'))
	shutil.rmtree(os.path.join(data_dir,'Training_yolo'))
		
if __name__ == '__main__':
	import sys
	#print(sys.argv[1])
	args = sys.argv[1:]
	_main_(args)