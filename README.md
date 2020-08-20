# PolEval_2020_task_4

To run this Project you will need:
- data from PolEval 2020 contest - task 4
- word embedding for Polish language - you can find it here: http://dsmodels.nlp.ipipan.waw.pl/
- Polish stopwords file
- Polish male and female names and surnames - available online: https://dane.gov.pl/

To use some of the models it is necessary to use Morfeusz package: http://morfeusz.sgjp.pl/. In used in code folders: train, validate, test there should be csv files which contains two columns, each file for each report, in the first column there should be words in forms that occur in the considered report, while in the second one an original/basic form of these words.
