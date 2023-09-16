from __future__ import annotations

import datetime
import decimal
import re
import warnings
from functools import partial
from typing import Union

import pyspark.sql.functions as F
import re2
from pyspark.sql import Column
from pyspark.sql.types import ArrayType, BooleanType, StringType

PrimitiveType = Union[bool, float, int, str]
DecimalLiteral = decimal.Decimal
DateTimeLiteral = Union[datetime.datetime, datetime.date]
LiteralType = PrimitiveType
ColumnOrName = Union[Column, str]

WARNING_STR = lambda func_name: f"Using {func_name} from regex_utils. Make sure your regex is Google-re2 or PCRE!"


def split(string: ColumnOrName, pattern: str, limit: int = -1) -> Column:
    """
    Splits string around matches of the given pattern.
    Uses Google-re2 for linear-time regexes, fallbacks to python exponential-time RE.

    Args:
        string (ColumnOrName): A string expression to split
        pattern (str): A string representing a regular expression.
        limit (int, optional): An integer which controls the number of times `pattern` is applied. Defaults to -1.
        * ``limit > 0``: The resulting array's length will not be more than `limit`, and the
                         resulting array's last entry will contain all input beyond the last
                         matched pattern.
        * ``limit <= 0``: `pattern` will be applied as many times as possible, and the resulting
                          array can be of any size.

    Returns:
        Column: Array of separated strings.
    """

    warnings.warn(WARNING_STR("split"), stacklevel=2)

    if limit <= 0:
        limit = 0

    try:
        split_partial = partial(re2.split, pattern=pattern, maxsplit=limit)
        res = F.udf(f=lambda string: split_partial(text=string), returnType=ArrayType(StringType(), False))(string)

    except:
        split_partial = partial(re.split, pattern=pattern, maxsplit=limit)
        res = F.udf(f=lambda string: split_partial(string=string), returnType=ArrayType(StringType(), False))(string)

    return res


def regexp_replace(string: ColumnOrName, pattern: Union[str, Column], replacement: Union[str, Column]) -> Column:
    """
    Replace all substrings of the specified string value that match regexp with replacement.

    Args:
        string (ColumnOrName): Column name or column containing the string value
        pattern (Union[str, Column]): Column object or str containing the regexp pattern
        replacement (Union[str, Column]): Column object or str containing the replacement

    Returns:
        Column: String with all substrings replaced.
    """

    warnings.warn(WARNING_STR("regexp_replace"), stacklevel=2)

    string = F.col(string) if not isinstance(string, Column) else string
    pattern = F.lit(pattern) if not isinstance(pattern, Column) else pattern
    replacement = F.lit(replacement) if not isinstance(replacement, Column) else replacement

    try:
        res = F.udf(f=lambda struct: re2.sub(pattern=struct["pattern"], repl=struct["replacement"], text=struct["string"]), returnType=StringType())(
            F.struct(string.alias("string"), pattern.alias("pattern"), replacement.alias("replacement"))
        )

    except:
        res = F.udf(f=lambda struct: re.sub(pattern=struct["pattern"], repl=struct["replacement"], string=struct["string"]), returnType=StringType())(
            F.struct(string.alias("string"), pattern.alias("pattern"), replacement.alias("replacement"))
        )

    return res


def regexp_extract(string: ColumnOrName, pattern: str, idx: int) -> Column:
    """
    Extract a specific group matched by the Java regex `regexp`, from the specified string column.
    If the regex did not match, or the specified group did not match, an empty string is returned.

    Args:
        string (ColumnOrName): Column name or column containing the string value
        pattern (str): A string representing a regular expression.
        idx (int): Matched group id.

    Returns:
        Column: Matched value specified by `idx` group id.
    """

    warnings.warn(WARNING_STR("regexp_extract"), stacklevel=2)

    try:

        def get_group(string: str):
            res = re2.search(pattern=pattern, text=string)

            if res is not None:
                return res.group(idx)

            else:
                return ""

        res = F.udf(f=get_group, returnType=StringType())(string)

    except:

        def get_group(string: str):
            res = re.search(pattern=pattern, string=string)

            if res is not None:
                return res.group(idx)

            else:
                return ""

        res = F.udf(f=get_group, returnType=StringType())(string)

    return res


def rlike(self: Column, other: str) -> Column:
    """
    SQL RLIKE expression (LIKE with Regex).
    Uses Google-re2 for linear-time regexes, fallbacks to python exponential-time RE.

    Args:
        other (str): An extended regex expression

    Returns:
        Column: Column of booleans showing whether each element in the Column is matched by extended regex expression.
    """

    warnings.warn(WARNING_STR("rlike"), stacklevel=2)

    try:
        search_partial = partial(re2.search, pattern=other)
        res = F.udf(f=lambda self: search_partial(text=self) is not None, returnType=BooleanType())(self)

    except:
        search_partial = partial(re.search, pattern=other)
        res = F.udf(f=lambda self: search_partial(string=self) is not None, returnType=BooleanType())(self)

    return res


def startswith(
    self: Column,
    other: Union[Column, LiteralType, DecimalLiteral, DateTimeLiteral],
) -> Column:
    """
    String starts with.
    Uses Google-re2 for linear-time regexes, fallbacks to python exponential-time RE.

    Args:
        other (Union[Column, LiteralType, DecimalLiteral, DateTimeLiteral]): String at start of line (do not use a regex `^`)

    Returns:
        Column: Column of booleans showing whether each element in the Column starts with.
    """

    warnings.warn(WARNING_STR("startswith"), stacklevel=2)

    other = F.lit(other) if not isinstance(other, Column) else other

    try:
        res = F.udf(
            f=lambda struct: re2.search(pattern=("^" + re2.escape(struct["other"])), text=struct["self"]) is not None, returnType=BooleanType()
        )(F.struct(self.alias("self"), other.alias("other")))

    except:
        res = F.udf(
            f=lambda struct: re.search(pattern=("^" + re.escape(struct["other"])), string=struct["self"]) is not None, returnType=BooleanType()
        )(F.struct(self.alias("self"), other.alias("other")))

    return res


def endswith(
    self: Column,
    other: Union[Column, LiteralType, DecimalLiteral, DateTimeLiteral],
) -> Column:
    """
    String ends with.
    Uses Google-re2 for linear-time regexes, fallbacks to python exponential-time RE.

    Args:
        other (Union[Column, LiteralType, DecimalLiteral, DateTimeLiteral]): String at end of line (do not use a regex `^`)

    Returns:
        Column: Column of booleans showing whether each element in the Column ends with.
    """

    warnings.warn(WARNING_STR("endswith"), stacklevel=2)

    other = F.lit(other) if not isinstance(other, Column) else other

    try:
        res = F.udf(
            f=lambda struct: re2.search(pattern=(re2.escape(struct["other"]) + "$"), text=struct["self"]) is not None, returnType=BooleanType()
        )(F.struct(self.alias("self"), other.alias("other")))

    except:
        res = F.udf(
            f=lambda struct: re.search(pattern=(re.escape(struct["other"]) + "$"), string=struct["self"]) is not None, returnType=BooleanType()
        )(F.struct(self.alias("self"), other.alias("other")))

    return res


# Monkey-patching all the functions
#
# This way, for example: If you call F.split, you will get our linear-time implementation instead of the native Java exponential-time implementation.
F.split = split
F.regexp_replace = regexp_replace
F.regexp_extract = regexp_extract
Column.rlike = rlike
Column.startswith = startswith
Column.endswith = endswith
