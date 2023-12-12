import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import streamlit as st
tokenizer = T5Tokenizer.from_pretrained("cointegrated/rut5-base-multitask")
model = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-base-multitask")


st.title('Сокращение текста')

text = st.text_area("Введите текст для сокращения:")

button = st.button('Сократить!')


def generate(text, **kwargs):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        hypotheses = model.generate(**inputs, num_beams=5, **kwargs)
    return tokenizer.decode(hypotheses[0], skip_special_tokens=True)


# text = """Файловая система связывает носитель информации, с одной стороны, и API для доступа к файлам, с другой. Когда прикладная программа обращается к файлу, она не имеет никакого представления о том, каким образом расположена информация в конкретном файле. Все, что знает программа — это имя файла, его размер и атрибуты. Эти данные она получает от драйвера файловой системы. Именно файловая система устанавливает, где и как будет записан файл на физическом носителе (например, жестком диске).
#
# С точки зрения операционной системы, весь диск представляет из себя набор кластеров размером от 512 байт. Драйверы файловой системы организуют кластеры в файлы и каталоги, реально являющиеся файлами, содержащими список файлов в этом каталоге. Эти же драйверы отслеживают, какие из кластеров в настоящее время используются, какие свободны, какие помечены как неисправные.
#
# Однако файловая система необязательно напрямую связана с физическим носителем информации. Существуют виртуальные и сетевые файловые системы, которые являются лишь способом доступа к файлам, находящимся на удалённом компьютере."""
#
# print(generate(f'simplify | {text}', max_length=200, length_penalty=3, no_repeat_ngram_size=3))

if button:
    result = generate(f'simplify | {text}', min_length=300, max_length=1024, length_penalty=3, no_repeat_ngram_size=3)
    st.write(result)
