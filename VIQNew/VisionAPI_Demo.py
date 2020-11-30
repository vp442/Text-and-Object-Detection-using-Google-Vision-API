import io, os
from google.cloud import vision
from google.cloud.vision_v1 import types
from draw_vertice import drawVertices
import pandas as pd
import random
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.corpus import stopwords 
from nltk.tokenize import  word_tokenize, RegexpTokenizer
from Pillow_Utility import draw_borders


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'

client = vision.ImageAnnotatorClient()
#client


###################################### Text from Image ###############################################

def detectText(img):
    with io.open(img, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content = content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    df = pd.DataFrame(columns=['locale','description'])
    for text in texts:
        df = df.append(dict(locale = text.locale, description = text.description), ignore_index = True)
    
    ############## Removing unwanted words ##############
    stop_words = set(stopwords.words('english')) 
    tokenizer = nltk.RegexpTokenizer(r"\w+") 
    
    df.drop(df.index[0], inplace = True)
    

    list_of_words = df['description'].to_list()
    print(list_of_words)
    print('\n')

    sentence = " "
    sentence = sentence.join(list_of_words)  #Converting list to string
    print(sentence)
    print('\n')

    
    word_tokens = word_tokenize(sentence) 
    print(word_tokens)
    print('\n') 
    
    filtered_sentence = [w for w in word_tokens if not w in stop_words]  
    filtered_sentence = []  
  
    for w in list_of_words:  
        if w not in stop_words:  
            filtered_sentence.append(w)
    without_punc = " "
    new_words = tokenizer.tokenize(without_punc.join(filtered_sentence))
    print(new_words)
    print(len(new_words))
    print('\n')

    pos_tags = nltk.pos_tag(new_words)
    

    chunks = nltk.ne_chunk(pos_tags, binary = False)
    for chunk in chunks:
        print(chunk)
    
    entities = []
    labels = []
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            entities.append(' '.join(c[0] for c in chunk))
            labels.append(chunk.label())
    
    entities_labels = list(set(zip(entities, labels)))
    entities_df = pd.DataFrame(entities_labels)
    entities_df.columns = ["Entities", "Labels"]
    print(entities_df)
    
    #print(filtered_sentence)
    print('\n')
    return df



FILE_NAME = 'print_text.jpg'
FOLDER_PATH = r'C:\\Users\\surab\\pyproj\\Vinits Shit\\VIQNew\\Images\\texts'
print(detectText(os.path.join(FOLDER_PATH, FILE_NAME)))



######################################### Image from URL ############################################

img_url = 'https://img.tfd.com/cde/_LSYJPG2.JPG'
image = types.Image()
image.source.image_uri = img_url
response = client.text_detection(image=image)
texts = response.text_annotations
df = pd.DataFrame(columns=['locale','description'])
for text in texts:
    df = df.append(dict(locale = text.locale, description = text.description), ignore_index = True)

stop_words = set(stopwords.words('english'))  
list_of_words = df['description'].to_list()
sentence = ""
sentence = sentence.join(list_of_words)  
word_tokens = word_tokenize(sentence)  
filtered_sentence = [w for w in word_tokens if not w in stop_words]  
filtered_sentence = []  
  
for w in word_tokens:  
    if w not in stop_words:  
        filtered_sentence.append(w)
print(filtered_sentence)
print(df)


#################################### Handwritten Text ########################################

def detectHandwriting(img):
    with io.open(img, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content = content)
    response = client.document_text_detection(image=image)
    docText = response.full_text_annotation.text
    return docText
    
FILE_NAME = 'print_text.jpg'
FOLDER_PATH = r'C:\\Users\\surab\\pyproj\\Vinits Shit\\VIQNew\\Images\\texts'

print(detectHandwriting(img = os.path.join(FOLDER_PATH, FILE_NAME)))



################################## Confidence Elements #########################################


FILE_NAME = 'handwritten1.jpg'
FOLDER_PATH = r'C:\\Users\\surab\\pyproj\\Vinits Shit\\VIQNew\\Images\\texts'
FILE_PATH = os.path.join(FOLDER_PATH, FILE_NAME)
with io.open(FILE_PATH, 'rb') as image_file:
    content = image_file.read()
image = types.Image(content = content)
response = client.document_text_detection(image=image)

pages = response.full_text_annotation.pages
for page in pages:
    for block in page.blocks:
        print('Block Confidence: ', block.confidence)

        for paragraph in block.paragraphs:
            print('Paragraph Confidence: ', paragraph.confidence)

            for word in paragraph.words:
                word_text = ''.join([symbol.text for symbol in word.symbols])

                print('Word text: {0} (confidence: {1}'.format(word_text, word.confidence))

                for symbol in word.symbols:
                    print('\tSymbol: {0} (confidence: {1}'.format(symbol.text, symbol.confidence))


######################################## Detect crop hints #######################################


def detect_crop_hints(path, aspect_Ratios):
    
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)

    crop_hints_params = vision.CropHintsParams(aspect_ratios=aspect_Ratios)
    image_context = types.ImageContext(
        crop_hints_params=crop_hints_params)

    response = client.crop_hints(image=image, image_context=image_context)
    hints = response.crop_hints_annotation.crop_hints

    for n, hint in enumerate(hints):
        print('\nCrop Hint: {}'.format(n))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in hint.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))


file_Path = '.\\Images\\texts\\image1.jpg'
aspect_Ratios = [16/9, 4/3]
detect_crop_hints(file_Path, aspect_Ratios)



######################################## Detect faces ##########################################

def detect_faces(path):
    
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)

    response = client.face_detection(image = image)
    faceAnnotation = response.face_annotations
    

    likelihood = ('Unknown', 'Very Unlikely', 'Unlikely', 'Possibly', 'Likely', 'Very Likely')

    print('Faces:')
    for face in faceAnnotation:
        print('Detection Confidence {0}'.format(face.detection_confidence))
        print('Angry Likelihood: {0}'.format(likelihood[face.anger_likelihood]))
        print('Joy Likelihood: {0}'.format(likelihood[face.joy_likelihood]))
        print('Sorrow Likelihood: {0}'.format(likelihood[face.sorrow_likelihood]))
        print('Surprised Likelihood: {0}'.format(likelihood[face.surprise_likelihood]))
        print('Headwear Likelihood: {0}'.format(likelihood[face.headwear_likelihood]))

        face_vertices = ['{0},{1}'.format(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        print('Face bound: {0}'.format(', '.join(face_vertices)))
        print('')


file_Path = '.\\Images\\texts\\Angry_image_2.jpg'
detect_faces(file_Path)



######################################## Detect dominant colors in the image ###################################

def detect_dominant_color(path):
    
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)

    response = client.image_properties(image = image).image_properties_annotation
    dominant_colors = response.dominant_colors

    for color in dominant_colors.colors:
        print('Pixel Fraction: {0}'.format(color.pixel_fraction))
        print('Score: {0}'.format(color.score))
        print('\tRed: {0}'.format(color.color.red))
        print('\tGreen: {0}'.format(color.color.green))
        print('\tBlue: {0}'.format(color.color.blue))
        print('')


file_Path = '.\\Images\\texts\\aqua_green_color.jpg'
detect_dominant_color(file_Path)



##################################### Detect Labels ###########################################

def detect_labels(path):
    
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)

    response = client.label_detection(image = image)
    labels = response.label_annotations

    df = pd.DataFrame(columns = ['description', 'score', 'topicality'])
    for label in labels:
        df = df.append(dict(description = label.description, score = label.score, topicality = label.topicality), ignore_index = True)
    return df


file_Path = '.\\Images\\texts\\Parrot_label.jpg'
print(detect_labels(file_Path))



##################################### Detect Famous Landmarks #######################################


def detect_landmark(path):
    
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)
    response = client.landmark_detection(image = image)
    landmarks = response.landmark_annotations

    df = pd.DataFrame(columns = ['description', 'locations', 'score'])
    for landmark in landmarks:
        df = df.append(dict(description = landmark.description, locations = landmark.locations, score = landmark.score), ignore_index = True)
    return df


file_Path = '.\\Images\\texts\\Eiffel_Tower.jpg'
print(detect_landmark(file_Path))



######################################## Detect logos ###########################################

def detect_logo(path):
    
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)
    response = client.logo_detection(image = image)
    logos = response.logo_annotations

    df = pd.DataFrame(columns = ['description', 'score'])
    for logo in logos:
        df = df.append(dict(description = logo.description, score = logo.score), ignore_index = True)
        vertices = logo.bounding_poly.vertices
        drawVertices(content, vertices, logo.description)
    return df


file_Path = '.\\Images\\texts\\apple-office.jpg'
print(detect_logo(file_Path))



######################################### Detect Objects ###########################################

def detect_object(path):
    
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)
    response = client.object_localization(image = image)
    localized_object_annotations = response.localized_object_annotations

    df = pd.DataFrame(columns = ['name', 'score'])
    for obj in localized_object_annotations:
        df = df.append(dict(name = obj.name, score = obj.score), ignore_index = True)

    pillow_image = Image.open(path)
    for obj in localized_object_annotations:
        r, g, b = random.randint(150, 255), random.randint(150, 255), random.randint(150,255)
        draw_borders(pillow_image, obj.bounding_poly, (r, g, b), pillow_image.size, obj.name, obj.score)
    pillow_image.show()
    
    return df


file_Path = '.\\Images\\texts\\ikea_object1.jpg'
print(detect_object(file_Path))



######################################################################

