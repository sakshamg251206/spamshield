# Dataset

`sms_spam.csv` is the **SMS Spam Collection v.1** by Tiago A. Almeida and José María
Gómez Hidalgo: 5,572 English SMS messages, each labelled `ham` (legitimate) or `spam`.

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Citation: Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. *Contributions to the Study of
  SMS Spam Filtering: New Collection and Results.* DocEng 2011.

Columns: `Category` (`ham`/`spam`) and `Message` (raw text). The file starts with a UTF-8
byte-order mark, which the loader handles.
