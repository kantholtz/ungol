[[_TOC_]]

```
pip install -r requirements.txt
pip install ipykernel
ipython kernel install --user --name=ungol-data
```


# Data Sets

Full data sets can be found elsewhere. Examples are part of this repository.

| ID            | Name                           | Year      | No      | Type    | Size   | Lang  |
|---------------|--------------------------------|-----------|---------|---------|--------|-------|
| CLEF-FR-9495] | CLEF: Frankfurter Rundschau    | 1994-1995 | 139.715 | docs    | 320 MB | DE    |
| CLEF-SP-9495] | CLEF: Der Spiegel              | 1994-1995 | 13.979  | docs    | 63 MB  | DE    |
| CLEF-SDA-94]  | CLEF: SDA                      | 1994      | 71.677  | docs    | 144 MB | DE    |
| CLEF-SDA-95]  | CLEF: SDA                      | 1995      | 69.438  | docs    | 141 MB | DE    |
| CLEF-GIRT-4   | CLEF: Social Sciences          | ?         | 302.638 | docs    | 524 MB | DE/EN |
| CLEF-GIRT-TH  | CLEF: Social Science Thesaurus | ?         | 10.624  | entries | 317 B  | DE/EN |
| CLEF-GIRT-TRT | CLEF: Translation Table        | ?         | 9.793   | entries | 160 B  | DE/RU |
| CLEF-GIRT-TR  | CLEF: Translation Update       | ?         | ?       | ?       | ?      | DE/RU |
| CLEF-WIKI     | CLEF: Wikipedia Dump           | 2006      | ?       | docs    | ?      | all   |


Notes:

- **CLEF-GIRT-4**: pseudo-English corpus which is in fact a translation of
  the German corpus into English (does not contain as much textual
  information as the German version)

- **CLEF-GIRT-TR**: seems to be an updated version of CLEF-GIRT-TRT


## CLEF

The CLEF Initiative (Conference and Labs of the Evaluation Forum,
formerly known as Cross-Language Evaluation Forum) is a self-organized
body whose main mission is to promote research, innovation, and
development of information access systems with an emphasis on
multilingual and multimodal information with various levels of
structure.

- http://www.clef-initiative.eu (nvrn, see mails for password)
- http://direct.dei.unipd.it/ (nvrn, shit-tier password)
- http://catalog.elra.info/en-us/

In general: FR/SP/SDA data sets are for "ad-hoc (ah)"-Tasks and GIRT
data sets are used by "domain specific (ds)"-Tasks.

- The abbreviation TEL stands for "The European Library".
- Samples only contain SPIEGEL articles


### CLEF-E0008 (2000-2003) ADHOC

- `Topic 1 - N Document`-Relations
- Binary label whether a document is relevant for the topic
- *The SDA dataset is necessary, too:*

``` fish
for f in CLEF200?_ah-mono-de.txt
  echo $f:
  cat $f | grep -vE '^FR' | grep -vE '^SPIEGEL' | sort | uniq | wc -l
  echo
end

# CLEF2000_ah-mono-de.txt:
# 11335

# CLEF2001_ah-mono-de.txt:
# 16726

# CLEF2002_ah-mono-de.txt:
# 19394

# CLEF2003_ah-mono-de.txt:
# 21534
```

#### Ground Truth Data

- CLEF2000_ah-mono-de
- CLEF2001_ah-mono-de
- CLEF2002_ah-mono-de
- CLEF2003_ah-mono-de

#### Data Sets

- CLEF-FR-9495
- CLEF-SP-9495
- CLEF-SDA-94 (!)
- CLEF-SDA-95 (!)


#### Examples

Topic

``` xml
<topic lang="de">
    <identifier>150-AH</identifier>
    <title>AI gegen Todesstrafe</title>
    <description>
        Finde Berichte über direkte Aktionen von Amnesty
        International gegen die Todesstrafe.
    </description>
    <narrative>
        Amnesty International widmet sich der weltweiten
        Abschaffung der Todesstrafe. Relevante Dokumente müssen spezifische
        Aktionen von AI gegen die Todesstrafe beschreiben.
    </narrative>
</topic>
```

Ground Truth

```
141-AH 0 FR940123-000022 0
141-AH 0 FR940206-000328 0
141-AH 0 FR940206-000527 0
141-AH 0 FR940206-001020 0
141-AH 0 FR940206-001294 0
141-AH 0 FR940206-001829 0
...
```

Data

``` xml
<DOC>
    <DOCNO>SPIEGEL9495-000003</DOCNO>
    <DOCID>SPIEGEL9495-000003</DOCID>
    <ACCOUNT>
        #0BASE#SV92FF#00011994000010000301
    </ACCOUNT>
    <OBJECT>
        SP
    </OBJECT>
    <ISSUE>
        1
    </ISSUE>
    <PAGE>
        3
    </PAGE>
    <DATE>
        03.01.1994
    </DATE>

    <TITLE>
        Hausmitteilung Betr.: Waldsterben
    </TITLE>

    <TEXT>

        Umweltpolitik, vor zwei Jahrzehnten fast noch ein Fremdwort, ist
        mittlerweile Gegenstand der Geschichtsschreibung, und der SPIEGEL hat
        darin seinen Standort. Das jngst im Beck Verlag erschienene "Jahrbuch
        ologie 1994" etwa kommt bei der Frage, wie die Klimaprobleme in der
        Bundesrepublik "zum Politikum" geworden sind, auf den SPIEGEL-Titel
        ber die Folgen des sauren Regens ("Der Wald stirbt") aus dem
        Jahre 1981. Dieser Beitrag - Start einer dreiteiligen, sper mit
        etlichen Umweltpreisen bedachten Serie - ist nach dem Urteil im
        Jahrbuch als die "Entdeckung des Waldsterbens" zu bewerten. Der
        Verfasser jener Serie, SPIEGEL-Ressortleiter Jochen Blsche, hat zwlf
        Jahre danach gemeinsam mit der Kollegin Sylvia Schreiber abermals
        Ursachen und Ausmader inzwischen weithin verdrgten Naturkatastrophe
        recherchiert: Warum ht, trotz Entschwefelung und Katalysator, das
        Siechtum an? Schuld ist, so Blsche, ein "bislang zu wenig beachteter
        Schadstoff", der "gleichsam ein zweites Waldsterben" bewirkt (Seite
        38).

    </TEXT>
</DOC>
```

### CLEF-E0036 (2004-2008) ADHOC

- 2004 - no german ground truth
- 2005 - no german ground truth
- 2006 - no german ground truth (only "robust")
- 2007 - no german ground truth
- 2008 - no ground truth at all (only "tel")

#### Ground Truth Data

- CLEF2006_ah.robust-mono-de (?)
- CLEF2008_ah.tel-mono-de (?)

#### Data Sets

- CLEF-FR-9495
- CLEF-SP-9495
- CLEF-SDA-94
- CLEF-SDA-95


### CLEF-E0037 (2004-2008)

#### Data Sets

- CLEF-GIRT-4
- CLEF-GIRT-TH
- CLEF-GIRT-TRT
- CLEF-GIRT-TR


### CLEF-E0038 (2003-2008)

- Question/Answering Task
- Wikipedia is a necessary additional source
- Q/A examples:
  - Who is the "Iron Chancellor"? Otto von Bismarck
  - What is Atlantis? Space Shuttle
  - Name all airports in London, England. Gatwick, Stansted, Heathrow, Luton and City.
- Test sets: 200 questions
- Answer does not need to be worded exactly but unccecassary information is penalized
- Answer can be copied from the source document (but may be enhanced)

#### Data Sets

- CLEF-FR-9495
- CLEF-SP-9495
- CLEF-SDA-94
- CLEF-SDA-95
- CLEF-WIKI-06




