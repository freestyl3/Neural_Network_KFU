from random import randint
from PIL import Image

def randomize_data(class_names, case='Lowercase'):
    path = f'./{case}/'
    path_to_save = f'./test_images/{case}/'
    for class_ in class_names:
        for i in range(10):
            img = Image.open(path + class_ + f'/{i}.jpg').convert('L')
            img = img.rotate(randint(-45, 45))
            img.save(path_to_save + class_ + f'/{i}.jpg', "jpeg")
