import xmlschema
from collections import defaultdict
import pandas as pd
import pickle
from pathlib import Path
from typing import List

XML_LANG = '@{http://www.w3.org/XML/1998/namespace}lang'

key_dict = {'dc:contributor': lambda key, l: parse_item_list(key, l, allow_not_lang=True),
            'dc:title': lambda key, l: parse_item_list(key, l, allow_lang=True,
                                                       allow_not_lang=True),
            'dc:creator': lambda key, l: parse_item_list(key, l, allow_not_lang=True),
            'dc:subject': lambda key, l: parse_item_list(key, l, allow_lang=True),
            'dc:source': lambda key, l: parse_item_list(key, l, allow_lang=True),
            'dc:description': lambda key, l: parse_item_list(key, l, allow_lang=True),
            'dc:publisher': lambda key, l: parse_item_list(key, l, allow_lang=True),
            'dc:date': lambda key, l: parse_item_list(key, l, allow_not_lang=True),
            'dc:type': lambda key, l: parse_item_list(key, l, allow_lang=True),
            'dc:format': lambda key, l: parse_item_list(key, l, allow_lang=True,
                                                        allow_not_lang=True),
            'dc:identifier': lambda key, l: parse_item_list(key, l, allow_lang=True,
                                                            allow_not_lang=True),
            'dc:coverage': lambda key, l: parse_item_list(key, l, allow_none=True, allow_lang=True,
                                                          allow_not_lang=True),
            'dc:language': lambda key, l: parse_item_list(key, l, allow_not_lang=True),
            'dc:rights': lambda key, l: parse_item_list(key, l, allow_lang=True)
            }


def get_xml_schema(schema_file: str) -> xmlschema.XMLSchema:
    """ Return an XMLSchema object defined in an .xsd file
        Note that if any sub-schema are referenced in the file,
        they should be contained in the same folder as the main schema
        file.
        :param schema_file: .xsd file defining the schema
        :returns: the xml schema
    """
    with open(schema_file) as f:
        schema = f.read()

    schema = xmlschema.XMLSchema(schema)
    print(schema)
    return schema


def validate_schema(schema: xmlschema.XMLSchema, xml_file: str) -> bool:
    """ Validate an xml file using a schema,
        printing True (valid) or False (invalid)
        to the console
        :param schema: XML schema to use to validate the file
        :param xml_file: filename of the file to validate
        :returns: True/False for valid/invalid
    """
    print('Checking schema valid?')
    is_valid = schema.is_valid(xml_file)
    print(is_valid)
    return is_valid


def parse_item_list(key: str, l: List[dict], allow_none=False, allow_lang=False,
                    allow_not_lang=False) -> dict:
    """ Parse a list of items for a given xml key
    e.g. [{'$':val}, ...]
    into a dictionary providing the keys that should be associated with those values
    and the values as a list (or None).
    e.g. {key: [val, ...]}
    Note that values with a language tag
    e.g. {'$':val, '@{http://www.w3.org/XML/1998/namespace}lang': 'en'}
    will be separated into columns with a suffix
    indicating the language:
    e.g. {key_en: [val, ...], key_fr: [val, ...], key: [val, ...]}
    And known "sub-keys" which are identified as prefixed in their values
    e.g. {'$': 'geoscanid:value'}
    will also be given their own columns
    e.g. {key_geoscanid: value}
    :param key: the XML key associated with the list that this item came from
    :param l: the list of items associated with that key
    :param allow_none: whether or not to allow the entire list to be None
    :param allow_lang: whether or not to allow for items with language tags
    :param allow_not_lang: whether or not to allow for items without language tags
    :return: a dictionary containing the key:[values] mappings for all items in the list
    where each item is associated with the appropriate new key
    """
    if allow_none:
        print(l)
        if len(l) == 1 and l[0] is None:
            return {key: None}

    ret_dict = defaultdict(list)

    for item in l:

        if item is None:
            continue

        if XML_LANG in item:
            if not allow_lang:
                raise ValueError(f'Should be no lang tag for key {key}')

            k, v = parse_lang_item(key, item)

            ret_dict[k].append(v)

        else:
            if not allow_not_lang:
                raise ValueError(f'Should be no items without lang tag for key {key}')

            try:
                k, v = parse_non_lang_item_with_subkey(key, item)
            except:
                k, v = parse_non_lang_item(key, item)

            ret_dict[k].append(v)

    return ret_dict


def parse_lang_item(key: str, item: dict):
    """ Given a single item, in a list of XML values, which contains a language key
        e.g. {'$': '123456', '@{http://www.w3.org/XML/1998/namespace}lang': 'en'}
        create the key:value mapping for the associated column, adding the language to the column
        name
        e.g. key_en:123456
        :param key: the XML key associated with the list that this item came from
        :param item: the list item
        :returns: k, v - the key with which this item should be associated and the item's value
    """
    lang = item[XML_LANG]

    if lang not in ['en', 'fr']:
        raise ValueError(f'unknown language: {lang}')

    k = f'{key}_{lang}'
    v = item['$'] if '$' in item else None
    return k, v


def parse_non_lang_item_with_subkey(key: str, item: dict):
    """ Given a single item, in a list of XML values, which does not contain a language key,
        but does contain a hidden "sub label"
        e.g. {'$': 'geoscanid:123456'}
        create the key:value mapping for the associated column, adding the subkey to the column
        name
        e.g. key_geoscanid:123456
        :param key: the XML key associated with the list that this item came from
        :param item: the list item
        :returns: k, v - the key with which this item should be associated and the item's value
    """
    k, v = item['$'].split(':')

    if k not in ['geoscanid', 'info']:
        raise ValueError()

    k = f'{key}_{k}'
    return k, v


def parse_non_lang_item(key: str, item: dict):
    """ Given a single item, in a list of XML values, which does not contain a language key,
        e.g. {'$': 'pdf'}
        create the key:value mapping for the associated column.
        e.g. key:pdf
        :param key: the XML key associated with the list that this item came from
        :param item: the list item
        :returns: k, v - the key with which this item should be associated and the item's value
    """
    k = f'{key}'
    v = item['$']

    return k, v


def convert_xml_data_to_dataframe(data_dict: dict):
    """
    Convert the contents of an xml file (as a dictionary, produced through
    e.g. schema.to_dict(), to a dataframe
    :param data_dict: dict containing the xml data
    :return: dataframe containing the xml data
    """
    df_list = []
    for i, item in enumerate(data_dict['item']):
        print(f'Item: {i}')
        item_dict = {}
        for key, value in item.items():
            print(f'  Key: {key}')
            print(value)

            ret_dict = key_dict[key](key, value)
            item_dict.update(ret_dict)

        df_list.append(item_dict)

    df = pd.DataFrame(df_list)

    return df


def convert_xml_to_dataframe_and_save(xml_schema: str, xml_file: str, validate: bool):
    """
    Convert an xml file to a dataframe and save that dataframe to a csv with the
    name xml_file_df.csv (where xml_file is the actual name of the file)
    Along the way, checkpoint a pkl file with the loaded xml data, but unconverted
    to a dataframe, with the name xml_file_dict.pkl (where xml_file is the actual name
    of the file)
    If the output csv file already exists, it will simply be loaded.
    :param xml_schema: filename of the XML schema .xsd
    :param xml_file: filename of the XML .xml data
    :param validate: whether or not to validate the schema (note, this is a long process
        for large files)
    :return: the converted dataframe, which is also saved to a csv file
    """

    xml_df_file = f'{xml_file}_df.csv'
    if Path(xml_df_file).exists():
        print('Loading existing df file...')
        return pd.read_csv(xml_df_file)

    schema = get_xml_schema(xml_schema)

    if validate:
        validate_schema(schema, xml_file)
    #
    print('Loading xml...')
    pickle_filename = f'{xml_file}_dict.pkl'
    if Path(pickle_filename).exists():
        print('File exists... reading from pkl...')
        with open(pickle_filename, 'rb') as f:
            data_dict = pickle.load(f)
    else:
        data_dict = schema.to_dict(xml_file, process_namespaces=False, lazy=True)
        print('...dumping to pkl...')
        with open(pickle_filename, 'wb') as f:
            pickle.dump(data_dict, f)

    print(data_dict.keys())

    print(f'N items: {len(data_dict["item"])}')

    print('Converting xml...')
    df = convert_xml_data_to_dataframe(data_dict)
    print(df.head())
    df.to_csv(xml_df_file)

    return df


if __name__ == "__main__":
    #schema_file = 'data/nrcan/geoscan_flex.xsd'
    schema_file = "/home/stefania/Downloads/geoscan.xsd"
    # xml_file = 'data/nrcan/GEOSCAN_OSDP_EXTRACT_updates-utf8.xml'
    #xml_file = '/mnt/projects/eai-nlp/data/nrcan/GEOSCAN-extract-20200107105330.xml'
    #xml_file = '/mnt/projects/eai-nlp/custom-work/nrcan/data/GEOSCAN-nickel.xml'
    #xml_file = '/mnt/projects/eai-nlp/custom-work/nrcan/data/GEOSCAN-extract-20200211144755.xml'
    xml_file = "/home/stefania/Downloads/EAITest.xml"

    df = convert_xml_to_dataframe_and_save(xml_schema=schema_file, xml_file=xml_file,
                                           validate=True)