def int_to_spanish(n):
    # Basic numbers
    basic_numbers = {
        0: "cero",
        1: "uno",
        2: "dos",
        3: "tres",
        4: "cuatro",
        5: "cinco",
        6: "seis",
        7: "siete",
        8: "ocho",
        9: "nueve",
        10: "diez",
        11: "once",
        12: "doce",
        13: "trece",
        14: "catorce",
        15: "quince",
        16: "dieciséis",
        17: "diecisiete",
        18: "dieciocho",
        19: "diecinueve",
        20: "veinte",
        21: "veintiuno",
        22: "veintidós",
        23: "veintitrés",
        24: "veinticuatro",
        25: "veinticinco",
        26: "veintiséis",
        27: "veintisiete",
        28: "veintiocho",
        29: "veintinueve",
    }

    # Tens from 30 to 100
    tens = {
        30: "treinta",
        40: "cuarenta",
        50: "cincuenta",
        60: "sesenta",
        70: "setenta",
        80: "ochenta",
        90: "noventa",
        100: "cien",
    }

    # Hundreds
    hundreds = {
        100: "cien",
        200: "doscientos",
        300: "trescientos",
        400: "cuatrocientos",
        500: "quinientos",
        600: "seiscientos",
        700: "setecientos",
        800: "ochocientos",
        900: "novecientos",
    }

    # Function to handle numbers from 1 to 999
    def convert_under_thousand(number):
        if number <= 29:
            return basic_numbers[number]
        elif number < 100:
            ten, unit = divmod(number, 10)
            ten *= 10
            return tens[ten] if unit == 0 else f"{tens[ten]} y {basic_numbers[unit]}"
        else:
            hundred, remainder = divmod(number, 100)
            hundred *= 100
            if remainder == 0:
                return hundreds[hundred]
            elif hundred == 100:
                return f"ciento {convert_under_thousand(remainder)}"
            else:
                return f"{hundreds[hundred]} {convert_under_thousand(remainder)}"

    # Handling numbers up to 999,999
    if n == 0:
        return basic_numbers[0]
    elif n < 1000:
        return convert_under_thousand(n)
    elif n < 1000000:
        thousand_part, remainder = divmod(n, 1000)
        thousand_word = convert_under_thousand(thousand_part) + " mil"
        return (
            f"{thousand_word} {convert_under_thousand(remainder)}"
            if remainder
            else thousand_word
        )

    # For numbers greater than 999,999
    return "Número demasiado grande"


# Test the function with a number in the hundred-thousands
int_to_spanish(123456)


def int_to_papiamentu(n):
    # Basic numbers
    basic_numbers = {
        0: "sero",
        1: "un",
        2: "dos",
        3: "tres",
        4: "kuater",
        5: "sinku",
        6: "seis",
        7: "shete",
        8: "ocho",
        9: "nuebe",
        10: "dies",
        11: "diesun",
        12: "diesdos",
        13: "diestres",
        14: "dieskuater",
        15: "diesinku",
        16: "diesseis",
        17: "diesshete",
        18: "diesocho",
        19: "diesnuebe",
        20: "binti",
    }

    # Tens from 30 to 100
    tens = {
        30: "trinta",
        40: "kuarenta",
        50: "sinkuenta",
        60: "sesenta",
        70: "shetenta",
        80: "ochenta",
        90: "nobenta",
        100: "shen",
    }

    # Function to handle numbers from 1 to 99
    def convert_under_hundred(number):
        if number <= 20:
            return basic_numbers[number]
        else:
            ten, unit = divmod(number, 10)
            ten *= 10
            if unit == 0 or ten == 20:
                return basic_numbers[ten] if ten in basic_numbers else tens[ten]
            else:
                return f"{tens[ten]} i {basic_numbers[unit]}"

    # Function to handle numbers from 100 to 999
    def convert_under_thousand(number):
        if number < 100:
            return convert_under_hundred(number)
        else:
            hundred, remainder = divmod(number, 100)
            hundred_word = (
                "shen" if hundred == 1 else f"{convert_under_hundred(hundred)}shen"
            )
            return (
                f"{hundred_word} {convert_under_hundred(remainder)}"
                if remainder
                else hundred_word
            )

    # Handling numbers up to 999,999
    if n < 1000:
        return convert_under_thousand(n)
    elif n < 1000000:
        thousand_part, remainder = divmod(n, 1000)
        thousand_word = (
            "mil"
            if thousand_part == 1
            else f"{convert_under_thousand(thousand_part)} mil"
        )
        return (
            f"{thousand_word} {convert_under_thousand(remainder)}"
            if remainder
            else thousand_word
        )

    # For numbers greater than 999,999
    return "Number ta hopi grandi"
