#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import sys
import glob
import time
import base64
import argparse
import multiprocessing
from ctypes import cast
from ctypes import cdll
from ctypes import c_int
from ctypes import c_ubyte
from ctypes import POINTER
from ctypes import c_char_p
try:
	from PIL import Image, ImageFile
	ImageFile.LOAD_TRUNCATED_IMAGES = True
except:
	print('Install Pillow with "pip3 install Pillow"')
	quit()



def check_subfolders(input_folder):
	entries = os.listdir(input_folder)
	subfolders = [os.path.join(input_folder, entry).replace('\\', '/') for entry in entries
                  if os.path.isdir(os.path.join(input_folder, entry))]
	if not subfolders:
		SavedFolder = os.path.dirname(input_folder)
		return [input_folder], SavedFolder # Explicitly return None if no subfolders
	return subfolders, input_folder


def generateHash(outputFolder, libName, imagePath):
	try:
		workerId = multiprocessing.current_process().name
		imageFile = Image.open(imagePath, mode="r")
		if imageFile.mode != 'RGB':
			imageFile = imageFile.convert(mode = 'RGB')
		libPhotoDNA = cdll.LoadLibrary(os.path.join(outputFolder, libName))

		ComputeRobustHash = libPhotoDNA.ComputeRobustHash
		ComputeRobustHash.argtypes = [c_char_p, c_int, c_int, c_int, POINTER(c_ubyte), c_int]
		ComputeRobustHash.restype = c_ubyte

		hashByteArray = (c_ubyte * 144)()
		ComputeRobustHash(c_char_p(imageFile.tobytes()), imageFile.width, imageFile.height, 0, hashByteArray, 0)

		hashPtr = cast(hashByteArray, POINTER(c_ubyte))
		hashList = [str(hashPtr[i]) for i in range(144)]
		hashString = ','.join([i for i in hashList])
		hashList = hashString.split(',')
		for i, hashPart in enumerate(hashList):
			hashList[i] = int(hashPart).to_bytes((len(hashPart) + 7) // 8, 'big')
		hashBytes = b''.join(hashList)
		with open(os.path.join(outputFolder, workerId + '.txt'), 'a', encoding = 'utf8') as outputFile:
			# outputFile.write('"' + imagePath + '","' + hashString + '"\n') # uncomment if you prefer base10 hashes
			outputFile.write('"' + imagePath + '","' + base64.b64encode(hashBytes).decode('utf-8') + '"\n')
	except Exception as e:
		print(e)


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='', help='The root for image folder')
parser.add_argument('--benign_func_test', action='store_true', help='only used for attack/benign0.sh')

args = parser.parse_args()

if __name__ == '__main__':
	## If generated from specific root (image folder)
	# root = '../attack/func_logs/coco_val_photodna/brightness'
	# root = '../train_verify/data/mnist/testing'
	# root = '../train_verify/data/celeba'
	# root = '../train_verify/data/celeba_random'
	# root = '../train_verify/data/cifar10_val'

	root = args.root
	benign_func_test = args.benign_func_test

	inputFolders, SavedFolder = check_subfolders(root)
	outputFolder = os.getcwd()
	if sys.platform == "windows":
		libName = 'PhotoDNAx64.dll'
	elif sys.platform == "darwin":
		libName = 'PhotoDNAx64.so'
	else:
		libName = 'PhotoDNAx64.dll'

	j = 0
	for inputFolder in inputFolders:
		startTime = time.time()
		print('Generating hashes for all images under ' + inputFolder)
		p = multiprocessing.Pool()
		print('Starting processing using ' + str(p._processes) + ' threads.')
		imageCount = 0
		images = glob.glob(os.path.join(inputFolder, '**', '*.jp*g'), recursive = True)
		images.extend(glob.glob(os.path.join(inputFolder, '**', '*.png'), recursive = True))
		images.extend(glob.glob(os.path.join(inputFolder, '**', '*.gif'), recursive = True))
		images.extend(glob.glob(os.path.join(inputFolder, '**', '*.bmp'), recursive = True))
		for f in images:
			imageCount = imageCount + 1
			p.apply_async(generateHash, [outputFolder, libName, f])
		p.close()
		p.join()

		allHashes = []
		for i in range(p._processes):
			try:
				workerId = 'SpawnPoolWorker-' + str(i + 1 +j)
				with open(os.path.join(outputFolder, workerId + '.txt'), 'r', encoding = 'utf8') as inputFile:
					fileContents = inputFile.read().splitlines()
				allHashes.extend(fileContents)
				os.remove(os.path.join(outputFolder, workerId + '.txt'))
				#print('Merged the ' + workerId + ' output.')
			except FileNotFoundError:
				print(workerId + ' not used. Skipping.')
				pass

		# with open(os.path.join(SavedFolder, f'{fileName}.csv'), 'a', encoding = 'utf8', errors = 'ignore') as f:
			# for word in allHashes:
			# 	f.write(str(word) + '\n')

		data = []
		for word in allHashes:
			# print(word)
			# Strip the extra quotes and split the string
			parts = word.strip('"').split('","')
			image_path = parts[0]
			hash_hex = parts[1]

			relative_path = os.path.relpath(image_path, '../train_verify/data')

			# Extract image number from path
			# image_number = int(os.path.splitext(os.path.basename(image_path))[0])

			# Decode the base64 hash to bytes, then to integers
			hash_bytes = base64.b64decode(hash_hex)
			hash_bin = [int(byte) for byte in hash_bytes]
			hash_bin = ' '.join(map(str, hash_bin))

			# Store in list as a dictionary
			if benign_func_test:
				data.append({
					'image': relative_path.replace("\\", "/"),
					'hash_bin': hash_bin,
					'hash_hex': hash_hex
				})

			else:
				data.append({
					'image': relative_path.replace("\\", "/"),
					'hash_hex': hash_hex
				})

		# Sort data by image number
		data.sort(key=lambda x: x['image'])
		fileName = inputFolder.split('/')[-1]
		# fileName = 'mnist-train'
		csv_file_path = os.path.join(SavedFolder, f'{fileName}.csv')
		with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
			writer = csv.writer(file)
			if benign_func_test:
				writer.writerow(['image', 'hash_bin', 'hash_hex'])
			for item in data:
				if benign_func_test:
					writer.writerow([item['image'], item['hash_bin'], item['hash_hex']])
				else:
					writer.writerow([item['image'], item['hash_hex']])


		print('Results saved into ' + os.path.join(SavedFolder, f'{fileName}.csv'))
		print('Generated hashes for ' + f'{imageCount:,}' + ' images in ' + str(int(round((time.time() - startTime)))) + ' seconds.')
		j+=20

