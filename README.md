# neuralSDF

# Установка
1. Клонировать репозиторий:X
   * git clone https://github.com/Lorent1/neuralSDF
   * cd neuralSDF 
2. Сабмодули:
   * git submodule init && git submodule update 

# Сборка (CPU):
Сборка, используя Cmake
   * mkdir build-cpu && cd cmake-cpu
   * cmake -DCMAKE_BUILD_TYPE=Release ..
   * make -j 8
Использовать Run (cpu) в vscode

# Функционал
   * Была реализована только версия на CPU, использован openMP для параллельного вычисления
   * Прямой проход SIREN
   * Визуализация (файлы 1 и 2)
   * Примеры изображений можно увидеть в папке images 

## Рекомендации
Не стоит использовать размер изображения свыше `512x512` для второй модели, так как рендеринг картинки будет занимать больше 5 минут
