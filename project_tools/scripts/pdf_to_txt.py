#!/usr/bin/env python
# coding: utf-8
# Copyright (C) 2021 ServiceNow, Inc.
#
""" Run PDF -> CSV conversion

"""
from pdfminer.layout import LAParams, LTTextBox, LTTextContainer, LTChar, LTTextBoxVertical, LTTextBoxHorizontal
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from tqdm import tqdm
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict
from filelock import FileLock
import os
import pathlib
import pandas as pd
import re
import errno
import signal

# In seconds. Start as low as 60, then increase to 200, 1000 etc.
TIMEOUT_TIME = 4000


class TimeoutException(Exception):
    def __init__(self, *args):
        Exception.__init__(self, *args)


class EmptyFileException(Exception):
    def __init__(self, *args):
        Exception.__init__(self, *args)


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutException(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


@timeout(TIMEOUT_TIME, "Timeout processing page")
def process_page_w_timeout(interpreter, page, device):
    try:
        interpreter.process_page(page)
        layout = device.get_result()
    except:
        layout = None
    return layout


def pprint(s, do_print=True):
    if do_print:
        print(s)


@timeout(TIMEOUT_TIME, "Timeout processing file")
def extract_using_pdfminer(p, output_txt, WRITE_TXT, N_PAGES):
    DO_PRINT = False    # N_PAGES is not None and N_PAGES != -1 or (not WRITE_TXT)

    df = defaultdict(list)

    if WRITE_TXT:
        fout = open(output_txt, "w")
    else:
        fout = None

    print(p)

    fp = open(p, 'rb')
    rsrcmgr = PDFResourceManager()
    laparams = LAParams(detect_vertical=True)
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    pages = PDFPage.get_pages(fp)

    try:
        _ = next(pages)
        # hacky
        pages = PDFPage.get_pages(fp)
    except PDFTextExtractionNotAllowed:
        # disable extractable check
        print(f"Warning: file deemed not extractable: {fp}. Trying to extract anyway...")
        pages = PDFPage.get_pages(fp, check_extractable=False)
    except StopIteration:
        raise EmptyFileException("No page extracted from file")
    except Exception as e:
        # some weird error may occur here, e.g. the file is not deemed pdf by pdfminer
        print(e)
        raise Exception("unhandled exception occurred!")

    failed_pages = defaultdict(list)

    then = datetime.now()

    for ipage, page in tqdm(enumerate(pages)):

        retval = process_page(ipage, page, DO_PRINT, WRITE_TXT, fout, interpreter, device)
        if retval is None:
            # page was rotated?
            page.rotate = 90
            retval = process_page(ipage, page, DO_PRINT, WRITE_TXT, fout, interpreter, device)

            if retval is None:
                print(f'Rotated page still returns nothing: {p} -- Pg No.{ipage}')
                failed_pages["file"].append(p)
                failed_pages["ipage"].append(ipage)
                failed_pages["page"].append(page)
                continue

        time_elapsed = datetime.now() - then

        timeout_sec = timedelta(seconds=TIMEOUT_TIME)

        print('Elapsed:', time_elapsed)
        if time_elapsed > timeout_sec:
            raise TimeoutException("Timeout processing file")

        for k, v in retval.items():
            df[k] += v

        pprint('-----------------------------------------------------------', DO_PRINT)

        if N_PAGES is not None and N_PAGES > -1:
            print(N_PAGES)
            if ipage > N_PAGES:
                print(f'hit page limit of {N_PAGES}')
                break

    if WRITE_TXT:
        fout.close()
    fp.close()
    df = pd.DataFrame(df)
    failed_pages = pd.DataFrame(failed_pages)
    return df, failed_pages


def process_page(ipage, page, DO_PRINT, WRITE_TXT, fout, interpreter, device):
    df = defaultdict(list)

    if WRITE_TXT:
        fout.write('------------------------------------------<NEWPAGE>\n')

    layout = process_page_w_timeout(interpreter, page, device)

    if layout is None:
        return None

    n_objs = len(layout)
    n_vert = 0
    n_horz = 0
    for lobj in layout:
        if isinstance(lobj, LTTextBoxVertical):
            n_vert += 1
        if isinstance(lobj, LTTextBoxHorizontal):
            n_horz += 1

    if n_vert > n_horz:
        print(("Rotated page found:", ipage, n_objs, n_vert, n_horz))
        return None

    for lobj in layout:
        if isinstance(lobj, LTTextContainer):
            x, y, text = lobj.bbox[0], lobj.bbox[3], lobj.get_text()

            pprint('%r, %s' % (str(lobj.bbox), str(type(lobj))), DO_PRINT)
            for text_line in lobj:
                pprint(text_line.get_text(), DO_PRINT)
                # for c in  text_line._objs:
                # print(c)
                # if isinstance(c, LTChar):
                # print("%s %s"%(c.fontname, c.adv))
                #    print()

                if WRITE_TXT:
                    fout.write(text_line.get_text())

        elif isinstance(lobj, LTTextBox):
            x, y, text = lobj.bbox[0], lobj.bbox[3], lobj.get_text()
            pprint('%r, LTTextBox' % (str(lobj.bbox)), DO_PRINT)
            pprint(text, DO_PRINT)

            if WRITE_TXT:
                fout.write(text)
        else:
            text = None
            pprint('%r, %s' % (str(lobj.bbox), str(type(lobj))), DO_PRINT)

        if WRITE_TXT:
            fout.write('<P>\n')

        df['obj_type'].append(str(type(lobj)))
        df['pg'].append(ipage)
        df['pos_x0'].append(lobj.bbox[0])
        df['pos_y0'].append(lobj.bbox[1])
        df['pos_x1'].append(lobj.bbox[2])
        df['pos_y1'].append(lobj.bbox[3])
        df['text'].append(text)

    return df


def process_files_in_dir(LOCAL_DIR, OUTPUT_DIR, N_FILES, n_pages, write_output, write_txt, regex_match, number_range=None, nfile_range=None):
    i_file = 0
    i_file_looked_at = -1
    all_data = []

    timeout_pdf_path = OUTPUT_DIR + "timeout_files.txt"
    print(f'Timeout pdfs stored in {timeout_pdf_path}')

    empty_pdf_path = OUTPUT_DIR + "empty_files.txt"
    print(f'Empty pdfs stored in {empty_pdf_path}')

    # create an empty file to store the time out pdfs
    with FileLock(timeout_pdf_path + ".lock"):
        print("Lock acquired.")
        open(timeout_pdf_path, 'a').close()

    for p in pathlib.Path(LOCAL_DIR).iterdir():
        i_file_looked_at += 1

        if number_range:
            if p.stem < number_range[0] or p.stem > number_range[1]:
                continue
            print(p.stem)
        if regex_match is not None:
            if not re.search(regex_match, str(p)):
                continue

        if p.is_file() and p.suffix == ".pdf":

            if nfile_range:
                if i_file_looked_at >= int(nfile_range[1]) or i_file_looked_at < int(nfile_range[0]):
                    continue

            output_txt = OUTPUT_DIR + p.stem + ".pdfminer_split.txt"
            output_csv = OUTPUT_DIR + p.stem + ".pdfminer_split.csv"
            output_failed = OUTPUT_DIR + p.stem + ".pdfminer_failed_pages.csv"
            print(output_txt)
            print(output_csv)

            if not os.path.exists(output_csv) or not write_output:
                print('running')

                try:
                    df, failed_pages = extract_using_pdfminer(p, output_txt, N_PAGES=n_pages, WRITE_TXT=write_txt)
                    if not failed_pages.empty:
                        failed_pages.to_csv(output_failed)
                    df['file'] = output_txt
                    if write_output:
                        df.to_csv(output_csv)

                except TimeoutException as e:
                    with FileLock(timeout_pdf_path + ".lock"):
                        print(f"Lock to {timeout_pdf_path} acquired.")
                        with open(timeout_pdf_path, "a") as f:
                            f.write(f'{p}\n')
                    print(f'{str(e)}: {p}')
                    continue

                except EmptyFileException as e:
                    with FileLock(empty_pdf_path + ".lock"):
                        print(f"Lock to {empty_pdf_path} acquired.")
                        with open(empty_pdf_path, "a") as f:
                            f.write(f'{p}\n')
                    print(f'{str(e)}: {p}')
                    continue

                except Exception as e:
                    print(e)
                    print(f'Warning: unhandled exception occurred!')

            else:
                if os.path.exists(output_csv):
                    print('File already exists. Skipping...')
                else:
                    print('not running')
                continue

        elif p.is_dir():
            # get all pdfs in the folder (and subfolders)
            print(f"Directory encountered: {p}")
            pdfs = p.glob("**/*.pdf")

            i_file_looked_at -= 1

            for pdf in pdfs:
                i_file_looked_at += 1

                if nfile_range:
                    if i_file_looked_at >= int(nfile_range[1]) or i_file_looked_at < int(nfile_range[0]):
                        continue

                pdf_name = pdf.relative_to(LOCAL_DIR)
                output_txt = OUTPUT_DIR + str(pdf_name.parent).replace('/',
                                                                       '__') + "__" + pdf_name.stem + ".pdfminer_split.txt"
                output_csv = output_txt + '.csv'

                print(output_txt)
                print(output_csv)

                if not os.path.exists(output_csv) or not write_output:
                    try: 
                        df, failed_pages = extract_using_pdfminer(pdf, output_txt, N_PAGES=n_pages, WRITE_TXT=write_output)
                        df['file'] = output_txt
                        df.to_csv(output_csv)

                    except TimeoutException as e:
                        with FileLock(timeout_pdf_path + ".lock"):
                            print(f"Lock to {timeout_pdf_path} acquired.")
                            with open(timeout_pdf_path, "a") as f:
                                f.write(f'{pdf}\n')
                        print(f'{str(e)}: {pdf}')
                        continue

                    except EmptyFileException as e:
                        with FileLock(empty_pdf_path + ".lock"):
                            print(f"Lock to {empty_pdf_path} acquired.")
                            with open(empty_pdf_path, "a") as f:
                                f.write(f'{pdf}\n')
                        print(f'{str(e)}: {pdf}')
                        continue

                    except Exception as e:
                        print(e)
                        print(f'Warning: unhandled exception occurred!')                        
                else:
                    print('not running')
                    continue

        else:
            print(f"Not a pdf file: {p}")
            continue

        i_file += 1
        if N_FILES is not None and N_FILES != -1:
            if i_file >= N_FILES:
                break


def process_files_in_list(LOCAL_DIR, OUTPUT_DIR, N_FILES, n_pages, write_output, write_txt, regex_match, pdf_ids=None, pdf_paths=None):

    if pdf_ids is None and pdf_paths is None:
        raise ValueError("Have to specify one of pdf_ids, pdf_paths")

    elif pdf_ids and pdf_paths:
        raise ValueError("Can only accept one of pdf_ids, pdf_paths")

    elif pdf_ids:
        pdf_paths = [pathlib.Path(LOCAL_DIR) / (id + ".pdf") for id in pdf_ids]

    else:
        pdf_paths = [pathlib.Path(path.strip()) for path in pdf_paths]

    i_file = 0

    timeout_pdf_path = OUTPUT_DIR + "timeout_files.txt"
    print(f'Timeout pdfs stored in {timeout_pdf_path}')

    empty_pdf_path = OUTPUT_DIR + "empty_files.txt"
    print(f'Empty pdfs stored in {empty_pdf_path}')

    # create an empty file to store the time out pdfs
    with FileLock(timeout_pdf_path + ".lock"):
        print("Lock acquired.")
        open(timeout_pdf_path, 'a').close()


    for p in pdf_paths:

        if regex_match is not None:
            if not re.search(regex_match, str(p)):
                continue


        output_txt = OUTPUT_DIR + p.stem + ".pdfminer_split.txt"
        output_csv = OUTPUT_DIR + p.stem + ".pdfminer_split.csv"
        output_failed = OUTPUT_DIR + p.stem + ".pdfminer_failed_pages.csv"
        print(output_txt)
        print(output_csv)

        if not os.path.exists(output_csv) or not write_output:
            print('running')

            try:
                df, failed_pages = extract_using_pdfminer(p, output_txt, N_PAGES=n_pages, WRITE_TXT=write_txt)
                if not failed_pages.empty:
                    failed_pages.to_csv(output_failed)
                df['file'] = output_txt
                if write_output:
                    df.to_csv(output_csv)

            except TimeoutException as e:
                with FileLock(timeout_pdf_path + ".lock"):
                    print(f"Lock to {timeout_pdf_path} acquired.")
                    with open(timeout_pdf_path, "a") as f:
                        f.write(f'{p}\n')
                print(f'{str(e)}: {p}')
                continue

            except EmptyFileException as e:
                with FileLock(empty_pdf_path + ".lock"):
                    print(f"Lock to {empty_pdf_path} acquired.")
                    with open(empty_pdf_path, "a") as f:
                        f.write(f'{p}\n')
                print(f'{str(e)}: {p}')
                continue

            except:
                continue

        else:
            if os.path.exists(output_csv):
                print('File already exists. Skipping...')
            else:
                print('not running')
            continue

        i_file += 1
        if N_FILES is not None and N_FILES != -1:
            if i_file >= N_FILES:
                break


if __name__ == "__main__":
    """
    python notebooks/pdf_to_txt.py --N_FILES -1 --WRITE_OUTPUT --LOCAL_DIR /nrcan_p2/data/01_raw/20201006/geoscan/raw/pdf --OUTPUT_DIR /nrcan_p2/data/02_intermediate/20201006/geoscan/pdf/v1_20201125 --NUMBER_RANGE 211864 211864
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--N_FILES', type=int,
                        help='maximum number of files to process (for debugging, -1 or not present to ignore')
    parser.add_argument('--N_PAGES', type=int,
                        help='maxmium number of pages to process per file (for debugging, -1 or not present to ignore)')
    parser.add_argument('--WRITE_OUTPUT', dest='WRITE_OUTPUT', action='store_true',
                        help='whether or not to write the output (both csv and txt)')
    parser.add_argument('--WRITE_TXT', dest='WRITE_TXT', action='store_true',
                        help='whether or not to write the output txt file')
    parser.add_argument('--FILE_REGEX', help="file to run exclusively")
    parser.add_argument('--NUMBER_RANGE', nargs="+", help="file number range")
    parser.add_argument('--NFILE_RANGE', nargs="+", help="nfile range")
    parser.add_argument('--PDF_IDS', nargs="+", help="file id numbers")
    parser.add_argument('--PDF_PATHS_FILE', help='file with list of pdf paths to be processed')
    parser.set_defaults(WRITE_OUTPUT=False)
    parser.set_defaults(WRITE_TXT=False)

    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('--LOCAL_DIR', help="directory for reading pdfs", required=True)
    requiredNamed.add_argument('--OUTPUT_DIR', help="output directory", required=True)
    args = parser.parse_args()

    N_FILES = args.N_FILES
    LOCAL_DIR = args.LOCAL_DIR
    N_PAGES = args.N_PAGES
    WRITE_OUTPUT = args.WRITE_OUTPUT
    WRITE_TXT = args.WRITE_TXT
    FILE_REGEX = args.FILE_REGEX
    OUTPUT_DIR = args.OUTPUT_DIR
    NUMBER_RANGE = args.NUMBER_RANGE
    PDF_IDS = args.PDF_IDS
    PDF_PATHS_FILE = args.PDF_PATHS_FILE
    NFILE_RANGE = args.NFILE_RANGE

    print(f'Page limit: {N_PAGES}')
    print(f'Write output: {WRITE_OUTPUT}')
    print(f'Write txt: {WRITE_TXT}')

    if (NUMBER_RANGE and PDF_IDS) or (NUMBER_RANGE and PDF_PATHS_FILE) or (PDF_PATHS_FILE and PDF_IDS):
        raise ValueError('Can only accept one of NUMBER_RANGE, PDF_IDS, PDF_LIST')

    if NUMBER_RANGE:
        print(f'File number range: {NUMBER_RANGE}')

    if NFILE_RANGE:
        print(f'File iter number ragne: {NFILE_RANGE}')

    if PDF_IDS:
        print(f'Number of pdf ids: {len(PDF_IDS)}')

    if PDF_PATHS_FILE:
        with open(PDF_PATHS_FILE, "r") as f:
            pdf_paths = f.readlines()
        print(f'Number of pdf paths: {len(pdf_paths)}')

    if N_PAGES is not None and N_PAGES != -1 and WRITE_OUTPUT:
        raise ValueError('cannot both test and write...')

    if not WRITE_OUTPUT and WRITE_TXT:
        raise ValueError('cannot write txt and not write output...')

    if PDF_IDS:
        # only process specified files
        process_files_in_list(LOCAL_DIR=LOCAL_DIR, OUTPUT_DIR=OUTPUT_DIR, N_FILES=N_FILES, n_pages=N_PAGES,
                             write_output=WRITE_OUTPUT, write_txt=WRITE_TXT, regex_match=FILE_REGEX,
                             pdf_ids=PDF_IDS)

    elif PDF_PATHS_FILE:
        # only process specified files in a given file
        process_files_in_list(LOCAL_DIR=LOCAL_DIR, OUTPUT_DIR=OUTPUT_DIR, N_FILES=N_FILES, n_pages=N_PAGES,
                              write_output=WRITE_OUTPUT, write_txt=WRITE_TXT, regex_match=FILE_REGEX,
                              pdf_paths=pdf_paths)

    else:
        # go through all files in the dir
        process_files_in_dir(LOCAL_DIR=LOCAL_DIR, OUTPUT_DIR=OUTPUT_DIR, N_FILES=N_FILES, n_pages=N_PAGES,
                             write_output=WRITE_OUTPUT, write_txt=WRITE_TXT, regex_match=FILE_REGEX,
                             number_range=NUMBER_RANGE,
                             nfile_range=NFILE_RANGE)



