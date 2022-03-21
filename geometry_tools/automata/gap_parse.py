import re

WHITESPACE = " \n\t"
MAX_ERRLEN = 100

class GAPInputException(Exception):
    pass

def parse_record(text):
    record = {}
    identifier_name = ""

    i = 0
    while i < len(text):
        c = text[i]
        if c == ')':
            i += 1
            break
        elif c == ',':
            identifier_name = ""
        elif c == ":" and text[i+1] == "=":
            value, offset = parse_contents(text[i + 2:])
            record[identifier_name] = value
            i += offset
        elif c not in WHITESPACE:
            identifier_name += c
        i += 1
    return (record, i + 1)


def literal_contents(content_str):
    if re.match(r"\d*\.\d+", content_str):
        return float(content_str)
    elif re.match(r"\d+", content_str):
        return int(content_str)
    else:
        return content_str

def parse_contents(text):
    content = ""
    for i, c in enumerate(text):
        if c == '"':
            content, offset = parse_quote(text[i + 1:])
            return (content, offset + i + 1)
        elif c == '[':
            content, offset = parse_list(text[i + 1:])
            return (content, offset + i + 1)
        elif c in WHITESPACE:
            continue
        elif c == 'r':
            if len(text) > i + 4 and text[i:i+4] == 'rec(':
                content, offset = parse_record(text[i+4:])
                return (content, offset + i + 4)
            else:
                content += c
        elif c in ",)":
            return (literal_contents(content), i + 1)
        else:
            content += c
    return (literal_contents(content), len(text))

def parse_list(text):

    interval = re.match(r"((-?\d+)\.\.(-?\d+)\])", text)

    if interval and int(interval.group(2)) <= int(interval.group(3)):
        current_list = range(int(interval.group(2)),
                             int(interval.group(3)) + 1)
        return (current_list, len(interval.group(1)))

    current_list = []
    content = ""
    i = 0
    while i < len(text):
        c = text[i]
        if c == '"':
            content, offset = parse_quote(text[i+1:])
            current_list.append(content)
            content = ""
            i += offset
        elif c == ',':
            if len(content) > 0:
                current_list.append(literal_contents(content))
            content = ""
        elif c == '[':
            newlist, offset = parse_list(text[i+1:])
            current_list.append(newlist)
            i += offset
        elif c == ']':
            if len(content) > 0:
                current_list.append(literal_contents(content))
            i += 1
            return (current_list, i)
        elif c not in WHITESPACE:
            content += c

        i += 1

    raise GAPInputException('Unclosed [: [%s'%text[:MAX_ERRLEN])

def parse_quote(text):
    #no escaped quotations
    close_quote = text.find('"')
    if close_quote != -1:
        return (text[:close_quote], close_quote + 1)
    else:
        raise GAPInputException('Unclosed ": "%s'%text[:MAX_ERRLEN])

def load_record_file(filename):
    with open(filename, 'r') as recfile:
        record, length = parse_record(recfile.read())
        return record
