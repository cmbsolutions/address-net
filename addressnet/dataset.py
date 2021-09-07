from typing import Optional, Union, Callable, List
from collections import OrderedDict

import random
import tensorflow as tf
import numpy as np
import string

from addressnet.typo import generate_typo

# Schema used to decode data from the TFRecord file
_features = OrderedDict([
    ('postalcode', tf.io.FixedLenFeature([], tf.string)),
    ('housenumber', tf.io.FixedLenFeature([], tf.int64)),
    ('houseletter', tf.io.FixedLenFeature([], tf.string)),
    ('housenumberext', tf.io.FixedLenFeature([], tf.string)),
    ('streetname', tf.io.FixedLenFeature([], tf.string)),
    ('streetname_short', tf.io.FixedLenFeature([], tf.string)),
    ('city', tf.io.FixedLenFeature([], tf.string))
])

# List of fields used as labels in the training data
labels_list = [
    'postalcode',  # 2
    'housenumber',  # 3
    'houseletter',  # 4
    'housenumberext',  # 5
    'streetname',  # 6
    'city'  # 8
]
# Number of labels in total (+1 for the blank category)
n_labels = len(labels_list) + 1

# Allowable characters for the encoded representation
vocab = list(string.digits + string.ascii_lowercase + string.punctuation + string.whitespace)


def vocab_lookup(characters: str) -> (int, np.ndarray):
    """
    Converts a string into a list of vocab indices
    :param characters: the string to convert
    # :param training: if True, artificial typos will be introduced
    :return: the string length and an array of vocab indices
    """
    result = list()
    for c in characters.lower():
        try:
            result.append(vocab.index(c) + 1)
        except ValueError:
            result.append(0)
    return len(characters), np.array(result, dtype=np.int64)


def decode_data(record: List[Union[str, int, float]]) -> Union[str, int, float]:
    """
    Decodes a record from the tfrecord file by converting all strings to UTF-8 encoding, and any numeric field with
    a value of -1 to None.
    :param record: the record to decode
    :return: an iterator for yielding the decoded fields
    """
    for item in record:
        try:
            # Attempt to treat the item in the record as a string
            yield item.decode("UTF-8")
        except AttributeError:
            # Treat the item as a number and encode -1 as None (see generate_tf_records.py)
            yield item if item != -1 else None


def labels(text: Union[str, int], field_name: Optional[str], mutate: bool = True) -> (str, np.ndarray):
    """
    Generates a numpy matrix labelling each character by field type. Strings have artificial typos introduced if
    mutate == True
    :param text: the text to label
    :param field_name: the name of the field to which the text belongs, or None if the label is blank
    :param mutate: introduce artificial typos
    :return: the original text and the numpy matrix of labels
    """

    # Ensure the input is a string, encoding None to an empty to string
    if text is None:
        text = ''
    else:
        # Introduce artificial typos if mutate == True
        text = generate_typo(str(text)) if mutate else str(text)
    labels_matrix = np.zeros((len(text), n_labels), dtype=np.bool)

    # If no field is supplied, then encode the label using the blank category
    if field_name is None:
        labels_matrix[:, 0] = True
    else:
        labels_matrix[:, labels_list.index(field_name) + 1] = True
    return text, labels_matrix


def random_separator(min_length: int = 1, max_length: int = 3, possible_sep_chars: Optional[str] = r",./\  ") -> str:
    """
    Generates a space-padded separator of random length using a random character from possible_sep_chars
    :param min_length: minimum length of the separator
    :param max_length: maximum length of the separator
    :param possible_sep_chars: string of possible characters to use for the separator
    :return: the separator string
    """
    chars = [" "] * random.randint(min_length, max_length)
    if len(chars) > 0 and possible_sep_chars:
        sep_char = random.choice(possible_sep_chars)
        chars[random.randrange(len(chars))] = sep_char
    return ''.join(chars)


def join_labels(lbls: [np.ndarray], sep: Union[str, Callable[..., str]] = " ") -> np.ndarray:
    """
    Concatenates a series of label matrices with a separator
    :param lbls: a list of numpy matrices
    :param sep: the separator string or function that returns the sep string
    :return: the concatenated labels
    """
    if len(lbls) < 2:
        return lbls

    joined_labels = None
    sep_str = None

    # if `sep` is not a function, set the separator (`sep_str`) to `sep`, otherwise leave as None
    if not callable(sep):
        sep_str = sep

    for lbl in lbls:
        if joined_labels is None:
            joined_labels = lbl
        else:
            # If `sep` is a function, call it on each iteration
            if callable(sep):
                sep_str = sep()

            # Skip zero-length labels
            if lbl.shape[0] == 0:
                continue
            elif sep_str is not None and len(sep_str) > 0 and joined_labels.shape[0] > 0:
                # Join using sep_str if it's present and non-zero in length
                joined_labels = np.concatenate([joined_labels, labels(sep_str, None, mutate=False)[1], lbl], axis=0)
            else:
                # Otherwise, directly concatenate the labels
                joined_labels = np.concatenate([joined_labels, lbl], axis=0)

    assert joined_labels is not None, "No labels were joined!"
    assert joined_labels.shape[1] == n_labels, "The number of labels generated was unexpected: got %i but wanted %i" % (
        joined_labels.shape[1], n_labels)

    return joined_labels


def join_str_and_labels(parts: [(str, np.ndarray)], sep: Union[str, Callable[..., str]] = " ") -> (str, np.ndarray):
    """
    Joins the strings and labels using the given separator
    :param parts: a list of string/label tuples
    :param sep: a string or function that returns the string to be used as a separator
    :return: the joined string and labels
    """
    # Keep only the parts with strings of length > 0
    parts = [p for p in parts if len(p[0]) > 0]

    # If there are no parts at all, return an empty string an array of shape (0, n_labels)
    if len(parts) == 0:
        return '', np.zeros((0, n_labels))
    # If there's only one part, just give it back as-is
    elif len(parts) == 1:
        return parts[0]

    # Pre-generate the separators - this is important if `sep` is a function returning non-deterministic results
    n_sep = len(parts) - 1
    if callable(sep):
        seps = [sep() for _ in range(n_sep)]
    else:
        seps = [sep] * n_sep
    seps += ['']

    # Join the strings using the list of separators
    strings = ''.join(sum([(s[0][0], s[1]) for s in zip(parts, seps)], ()))

    # Join the labels using an iterator function
    sep_iter = iter(seps)
    lbls = join_labels([s[1] for s in parts], sep=lambda: next(sep_iter))

    assert len(strings) == lbls.shape[0], "string length %i (%s), label length %i using sep %s" % (
        len(strings), strings, lbls.shape[0], seps)
    return strings, lbls


def choose(option1: Callable = lambda: None, option2: Callable = lambda: None):
    """
    Randomly run either option 1 or option 2
    :param option1: a possible function to run
    :param option2: another possible function to run
    :return: the result of the function
    """
    if random.getrandbits(1):
        return option1()
    else:
        return option2()


def synthesise_address(*record) -> (int, np.ndarray, np.ndarray):
    """
    Uses the record information to construct a formatted address with labels. The addresses generated involve
    semi-random permutations and corruptions to help avoid over-fitting.
    :param record: the decoded item from the TFRecord file
    :return: the address string length, encoded text and labels
    """
    fields = dict(zip(_features.keys(), decode_data(record)))

    print(fields['streetname'] + ' ' + str(fields['housenumber']) + ' ' + fields['houseletter'] + fields[
        'housenumberext'] + ' ' + fields['postalcode'] + ' ' + fields['city'])

    street_number = generate_street_number(fields['housenumber'], fields['houseletter'], fields['housenumberext'])
    street = generate_street_name(fields['streetname'], fields['streetname_short'])
    postcode = labels(fields['postalcode'], 'postalcode')
    city = labels(fields['city'], 'city')
    # Begin composing the formatted address, building up the `parts` variable...

    city_postcode = list()
    # Keep city?
    # choose(lambda: city_postcode.append(city))
    # Keep postcode?
    # choose(lambda: city_postcode.append(postcode))
    city_postcode.append(postcode)
    city_postcode.append(city)
    random.shuffle(city_postcode)

    # parts = [[street], [street_number], [city_postcode]]
    parts = [[street, street_number]]

    random.shuffle(parts)

    parts.append(city_postcode)

    # Flatten the address components into an unnested list
    parts = sum(parts, [])

    # Join each address component/label with a random separator
    address, address_lbl = join_str_and_labels(parts, sep=lambda: random_separator(1, 3, possible_sep_chars=None))

    # Encode
    length, text_encoded = vocab_lookup(address)
    return length, text_encoded, address_lbl


def generate_street_number(housenumber: int, houseletter: str, housenumberext: str) -> (str, np.ndarray):
    """
    Generates a street number using the prefix, suffix, first and last number components
    :param housenumberext:
    :param houseletter:
    :param housenumber:
    :return: the street number
    """

    housenumber = labels(housenumber, 'housenumber')
    houseletter = labels(houseletter, 'houseletter')
    housenumberext = labels(housenumberext, 'housenumberext')

    return join_str_and_labels([housenumber, houseletter, housenumberext],
                               sep=random_separator(0, 2, possible_sep_chars=None))


def generate_street_name(streetname: str, streetname_short: str) -> (str, np.ndarray):
    """
   Generates a possible street name variation
   :param streetname: the street's name
   :param streetname_short: the street's short name
   :return: string and labels
   """

    if streetname_short is None or streetname_short == '':
        street_name = choose(lambda: labels(streetname, 'streetname'), lambda: labels(streetname, 'streetname'))
    else:
        street_name = choose(lambda: labels(streetname, 'streetname'), lambda: labels(streetname_short, 'streetname'))

    street_name_label = labels(streetname, 'streetname')

    return join_str_and_labels([street_name, street_name_label])


def dataset(filenames: [str], batch_size: int = 10, shuffle_buffer: int = 1000, prefetch_buffer_size: int = 10000,
            num_parallel_calls: int = 8) -> Callable:
    """
    Creates a Tensorflow dataset and iterator operations
    :param filenames: the tfrecord filenames
    :param batch_size: training batch size
    :param shuffle_buffer: shuffle buffer size
    :param prefetch_buffer_size: size of the prefetch buffer
    :param num_parallel_calls: number of parallel calls for the mapping functions
    :return: the input_fn
    """

    def input_fn() -> tf.data.Dataset:
        ds = tf.data.TFRecordDataset(filenames, compression_type="GZIP")
        ds = ds.shuffle(buffer_size=shuffle_buffer)
        ds = ds.map(lambda record: tf.io.parse_single_example(serialized=record, features=_features),
                    num_parallel_calls=8)
        ds = ds.map(
            lambda record: tf.compat.v1.py_func(synthesise_address, [record[k] for k in _features.keys()],
                                                [tf.int32, tf.int64, tf.bool],
                                                stateful=False),
            num_parallel_calls=num_parallel_calls
        )

        ds = ds.padded_batch(batch_size, ([], [None], [None, n_labels]))

        ds = ds.map(
            lambda _lengths, _encoded_text, _labels: ({'lengths': _lengths, 'encoded_text': _encoded_text}, _labels),
            num_parallel_calls=num_parallel_calls
        )
        ds = ds.prefetch(buffer_size=prefetch_buffer_size)
        return ds

    return input_fn


def predict_input_fn(input_text: List[str]) -> Callable:
    """
    An input function for one prediction example
    :param input_text: the input text
    :return:
    """

    def input_fn() -> tf.data.Dataset:
        predict_ds = tf.data.Dataset.from_generator(
            lambda: (vocab_lookup(address) for address in input_text),
            (tf.int64, tf.int64),
            (tf.TensorShape([]), tf.TensorShape([None]))
        )
        predict_ds = predict_ds.batch(1)
        predict_ds = predict_ds.map(
            lambda lengths, encoded_text: {'lengths': lengths, 'encoded_text': encoded_text}
        )
        return predict_ds

    return input_fn
