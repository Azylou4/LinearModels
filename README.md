# Анализ линейных моделей и проверка гипотез

## Описание проекта

Этот проект посвящен построению и анализу моделей линейной регрессии для оценки влияния различных предикторов на популярность песен. Работа включает:

1. Построение модели линейной регрессии с указанными параметрами.
2. Вычисление коэффициента детерминации.
3. Проверку гипотез при уровне значимости α = 0.05 (формулировка основной и альтернативной гипотез, расчет статистики критерия, вычисление критических значений, указание p-value).

## Датасет

- **Зависимая переменная**: популярность песни
- **Предикторы**: продолжительность песни, "танцевальность", "энергичность"

## Проверка гипотез

1. Чем больше "энергичность", тем больше популярность.
2. Популярность зависит от "танцевальности".
3. Популярность зависит одновременно от продолжительности и "танцевальности".

## Файлы в репозитории

- `song_data.csv`: Датасет с данными о песнях.
- `popularity_analysis.py`: Python-скрипт с реализацией анализа.

## Результаты анализа

Результаты включают коэффициенты модели, коэффициент детерминации, p-value для каждого предиктора, а также проверку статистических гипотез.
[Вывод программы (pdf)]()

## Решение в Google Colab

Вы также можете посмотреть и выполнить решение в Google Colab по [этой ссылке](https://colab.research.google.com/drive/1WH497rsf-q5vNLegzY7ONyT9U7jfOhxN?usp=sharing).


