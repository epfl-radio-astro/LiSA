class Tools:
    # Colours your terminal text for more impactfull information.
    # based on: https://www.geeksforgeeks.org/print-colors-python-terminal/
    # and: https://stackoverflow.com/questions/8924173/how-do-i-print-bold-text-in-python
    def tcol(text, colour):
        # We check for colours
        colour = colour.lower()
        if colour == 'red': col = '\033[91m'
        elif colour == 'green': col = '\033[92m'
        elif colour == 'yellow': col = '\033[93m'
        elif colour == 'lightpurple': col = '\033[94m'
        elif colour == 'purple': col = '\033[95m'
        elif colour == 'cyan': col = '\033[96m'
        elif colour == 'lightgray': col = '\033[97m'
        elif colour == 'black': col = '\033[98m'
        else: return text

        # # We take into account the bold and underline.
        # add = ''
        # if bold: add += '\033[1m'
        # if underline: add += '\033[4m'

        # We return the result
        return '{} {} \033[00m'.format(col, text)


        