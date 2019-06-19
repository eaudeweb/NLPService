import itertools
import logging
import re
from pathlib import Path

import click
import ftfy
import syntok.segmenter as segmenter
import textacy as tc
from nltk.corpus import stopwords

from .utils import get_es_records

logger = logging.getLogger(__name__)

MAL_URL_RE = 'hxxp:\/\/(.+)[\b|\s|$]'       # TODO: needs improvements
SHA256_RE = '[A-Fa-f0-9]{64}'
SHA1_RE = '[A-Fa-f0-9]{40}'
MD5_RE = '[A-Fa-f0-9]{32}'

N_RE = '\d{5,100}'

VERS_RE = [
    '\d{2}\.\d{3}\.\d{5}',  # 15.006.30119
    '\d{2}\.\d\.\d{2}',     # 11.0.14
    '\d+\.\d\.\d+'          # 11.0.14
]

# KnowledgeBase number such as KB4493730
KB_RE = re.compile(r'KB\d+')

# MS Security buletin such as MS16-010
MS_RE = re.compile(r'MS\d{2}-\d{3,5}')

CVE_RE = re.compile(r'CVE-\d{4}-\d{4,7}')


stops = set(stopwords.words('english'))


def remove_stopwords(tokens):
    return [x for x in tokens if x not in stops]


def fix_versions(text):
    for r in VERS_RE:
        text = re.sub(r, 'VERSIONID', text)

    return text


def fix_mal_url(text):
    # TODO: this is simplistic, should copy regexp from textacy

    return re.sub(MAL_URL_RE, 'MALURL', text)


def fix_sha(text):
    text = re.sub(SHA256_RE, 'MD5HASHID', text)
    text = re.sub(SHA1_RE, 'MD5HASHID', text)
    text = re.sub(MD5_RE, 'MD5HASHID', text)

    return text


def num_match(m):
    g = m.group(0)

    return 'DIGNR DIG{}ITS'.format(len(g))


def fix_numbers(text):
    text = re.sub(N_RE, num_match, text)

    return text

# terms copy/pasted from:
# https://support.microsoft.com/en-us/help/242450/how-to-query-the-microsoft-knowledge-base-by-using-keywords-and-query


_MS_TERMS = """kbBackup	Creating or restoring backups of the computer system
kbClustering	Server Cluster
kbDefrag	Microsoft Defragmenter
kbDisasterRec	Disaster recovery
kbDualBoot	Dual booting
kbenable	Accessibility: information
kbEnableHear	Accessibility: hearing features
kbEnableLearn	Accessibility: learning features
kbEnableMove	Accessibility: movement features
kbEnableSight	Accessibility: sight features
kbenv	Environment and configuration information including OS, registry
kbEvent	Events, event models, custom events
kbEventLog	Event logging
kbFax	Fax
kbformat	Formatting
kbkern32dll	Kernel 32 issues and errors in Windows
kbKernBase	Kernel/Base issues
kbNLB	Network Load Balancing Cluster, Windows Load Balancing Service (WLBS)
kbRD	Remote Desktop
kbRegistry	Windows registry
kbRPC	Remote procedure calls
kbSafeMod	Safe Mode
kbScanDisk	Microsoft ScanDisk
kbshell	The Windows shell that lets users group-start and otherwise control
kbsound	Sound/audio
kbSysPrepTool	Microsoft Systems Preparation Tool
kbSysSettings	Operating system setup and registry settings
kbTimeServ	W32 time service
kbVirtualMem	Virtual memory
kbwindowsupdate	Windows Update
kbWinsock	Communication sockets such as NET Winsock and MFC Socks
kbAppCompatibility	Application compatibility
kbConfig	Configuration after initial setup
kbdisplay	Displaying information on a monitor
kberrmsg	Error messages and error message follow-up information
kbgraphic	Graphics
kbHardware	Hardware
kbinterop	Operation between applications
kbmm	Multimedia
kbMobility	Mobile Information Server and ActiveSync
kbprint	Printing
kbproof	Spelling and grammar checking tools
kbRepair	Repair processes and tools
kbsetup	Setup
kbui	Configuring the user interface
kbUpgrade	Upgrading or migration
kbUSB	Universal serial bus
kbvirus	Viruses and macro viruses
kbwizard	Using wizards
atDownload	Contains a software update download
Kbqfe	Contains a hotfix
kbhowto	Describes a feature or describes how to perform a task
kbtshoot	Describes a problem or bug, how to fix a problem or bug, or for
KbSECBulletin	Security bulletins
kbSecurity	Security
KbSECVulnerability	Known software vulnerabilities
kbCookie	Browser or operating system cookies
kbFTP	File Transfer Protocol
kbHttpRuntime	HTTP Runtime
kburl	Contains a link to Internet Web site
kbWebBrowser	WebBrowser
kbActiveDirectory	Windows NT Active Directory
kbDCPromo	DCPromo.exe (the promotion process on a domain controller)
kbDHCP	Dynamic Host Configuration Protocol
kbDNS	Domain Name System (DNS)
kbFirewall	Firewalls
kbGPO	Group Policy objects
kbGRPPOLICYinfo	Group Policy information
kbGRPPOLICYprob	Group Policy problem
kbnetwork	Networking
kbStorageMgmt	Storage management
kbTermServ	Terminal Server
kbExchangeOWA	Outlook Web Access integrated with Exchange
kbTransport	SMTP and MTA
kbaddressbook	Address book issues
kbreceivemail	email receipt issues
kbsendmail	email sending issues
kbemail	General email category
kbBTSmessaging	Microsoft Biz Talk Server messaging
kbDatabase	Database
kbDTC	Microsoft Distributed Transaction Coordinator (DTC)
kbJET	JET database
kbOracle	Oracle products and technologies
kbActivexEvents	COM Events
kbActiveXScript	ActiveX Scripting
kbAPI	Application Programming Interface (API), must have another keyword in
kbCompiler	Compiler
kbCOMPlusLCE	COM+ Loosely Coupled Events
kbComPlusQC	COM+ Queued Components
kbCtrl	Programming or use of OCX or Windows controls
kbDCOM	Content related to Distributed Component Object Model
kbDirect3dIM	Direct3D-Immediate mode
kbDirect3dRM	Direct3D-Retained mode
kbDirectDraw	Direct Draw APIs
kbide	Integrated Development Environment (IDE)
kbJAFC	Java Application Foundation Classes
kbJava	Java programming/usage
kbJavaFAQ	Java Technologies frequently asked questions
kbJDBC	Java Database Connectivity
kbJNative	Native Java method
kbmacro	About an issue with the macro recorder or includes steps that use the
kbMSHTML	Microsoft HTML Rendering Control
kbProgramming	Programming
kbRemoting	.NET Framework remoting namespace objects and classes
kbSample	Contains compilable code samples
kbWebServices	Microsoft Web services
kbXML	XML
Vfw	Microsoft Video for Windows
Wps	Microsoft Windows Printing System
MSCS	Server Cluster
Wss	Microsoft Windows Sound System
Msvoice	Microsoft Voice
kbScanDisk	Microsoft ScanDisk
kbDiskMemory	Disk/Memory Management
ISVCompress	Third-party disk compression
LFN	Long file name
MWAV	Microsoft AntiVirus for Windows
MWBackup	Microsoft Backup for Windows
AWFax	Microsoft At Work PC Fax
RAS	Remote Access Services
WinComm	Windows communications/serial
WinDrvr	Windows driver-related information
WinMem	Windows memory
WinShell	Windows Shell issues
WinMIDI	MIDI
WinTTF	TrueType font issues
WinPNP	Plug and Play issues
WinTAPI	Telephony issues
WinPlus	Microsoft Plus!
Dun	Dial-Up Networking
Drvspace	DriveSpace
WinFat32	FAT32
WinGame	Games that are included with Windows
SysInfo	Microsoft System Information Tool issues
Multimon	Multiple Monitor issues
WinAPM	Advanced Power Management (APM) issues
WinBatch	Batch files
WinBoot	Windows Startup issues
Winprint	Microsoft Windows Printing issues
Netclients	Network client
NLB	Network Load Balancing Cluster, Windows Load Balancing Service (WLBS)
kbGPF	General Protection Faults and general crashes
UAE	Unrecoverable application error
3rdPartyNet	Third-party network issues
MSNets	Microsoft network issues
NetUtils	Network protocol utilities
NDIS3x	NDIS 3.x protocol issues
NDIS2	NDIS 2 protocol issues
NETHW	Network hardware issues
AppsComp	Application compatibility issues
msnsignup	Sign-up issues
msnconnect	Connection modem and access issues
msnbbs	Bulletin board issues
msnchat	Chat issues
msnmail	E-mail issues
msnnav	Navigation issues
msninterop	Interaction with Windows 95 issues
msnsetup	Setup and upgrade issues
msnaccount	Customer account issues
msninternet	Internet issues
msnother	Issues that are not covered by other MSN keywords
msn_encarta	Information that is related to Encarta on MSN
msn_bookshelf	Information that is related to Bookshelf on MSN"""

MS_TERMS = [t.split('\t')[0].strip() for t in _MS_TERMS.split('\n')]


def fix_kb(text):
    # TODO: better handling of MS_TERMS

    for t in MS_TERMS:
        text = text.replace(t, 'MSQUERYTERM')
    text = re.sub(KB_RE, 'MSKBID', text)
    text = re.sub(MS_RE, 'MSBULLETINID', text)

    return text


def fix_cve(text):
    return re.sub(CVE_RE, 'CVEID', text)


def fix_sentences(text):
    out = ''

    for paragraph in segmenter.process(text):
        for sentence in paragraph:
            for token in sentence:
                # roughly reproduce the input,
                # except for hyphenated word-breaks
                # and replacing "n't" contractions with "not",
                # separating tokens by single spaces
                out += token.value + ' '
            out += '\n'  # print one sentence per line
        out += '\n'

    return out


def fix_acronyms(text):
    # TODO: should remove the initial definition.
    # for ex: Indicator of Compromise (IoC) gets translated into
    # IndicatorOfCompromise (IndicatorOfCompromise)
    # probably this can be fixed by replacing "full", but needs window check
    doc = tc.make_spacy_doc(text)
    acros = tc.extract.acronyms_and_definitions(doc)

    for short, full in acros.items():
        label = "".join([t.title() for t in full.split(' ')])
        text = text.replace(full, label)
        text = text.replace(short, label)

    return text


def text_tokenize(text):
    """ breaks down the text to basic bits for a space-based tokenizer

    :param text: a text document
    :returns: a list of sentences, each its own list of tokens
    """

    text = clean(text)
    text = text.lower()
    text = tc.preprocess.remove_punct(text)
    text = tc.preprocess.replace_numbers(text, 'dignr')

    sentences = []

    for line in text.splitlines():
        tokens = remove_stopwords([t for t in line.split(' ') if len(t) > 1])

        if tokens:
            sentences.append(tokens)

    return sentences


FILTERS = [
    # fix_acronyms,
    tc.preprocess.normalize_whitespace,
    tc.preprocess.replace_urls,
    tc.preprocess.replace_emails,
    tc.preprocess.replace_phone_numbers,
    tc.preprocess.replace_currency_symbols,
    fix_mal_url,
    fix_sha,
    fix_cve,
    fix_kb,
    fix_versions,
    fix_numbers,
    ftfy.fix_text,
    fix_sentences,
]


def clean(text, filters=FILTERS):
    for f in filters:
        text = f(text)

    return text


@click.command()
@click.argument('output')
@click.option('--es-url',
              default='http://elasticsearch:9200/content',
              help='ElasticSearch index URL location')
@click.option('--count',
              default=0,
              help='How many documents to process')
def main(output, es_url, count):
    """ Download and clean documents from ES, writes cleaned to output location

    Output is a text file, with one sentence per line
    """

    logging.basicConfig()
    logger.setLevel(logging.INFO)

    fields = ['title', 'content']
    docs = get_es_records(es_url)

    if count:
        docs = itertools.islice(docs, 0, count)

    out = Path(output)

    i = 0
    with out.open('w') as outf:
        for rec in docs:
            i += 1
            texts = [rec[f].strip() for f in fields]
            doc = ".\n".join(texts)

            lines = [" ".join(sent) for sent in text_tokenize(doc)]
            outf.write("\n".join(lines))
            outf.write('\n\n')

    logger.info("Processed %s documents", i)

    return str(out)
