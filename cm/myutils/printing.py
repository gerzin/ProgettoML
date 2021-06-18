# colors
RED = "\033[1;31m"
BLUE = "\033[1;34m"
CYAN = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD = "\033[;1m"
REVERSE = "\033[;7m"


def print_decreasing(ite, v, g_norm, d_norm, old=None):
    formatter = ""
    if old is None:
        formatter = "%5d\t{}%1.16e\t{}%1.16e\t{}%1.16e\033[0;0m".format(
            GREEN, GREEN, GREEN)
    else:
        new = (ite, v, g_norm, d_norm)
        values = []
        for i in range(4):
            if old[i] - new[i] > 0:
                values.append(GREEN)
            else:
                values.append(RED)
        formatter = "%5d\t{0}%1.16e\t{1}%1.16e\t{2}%1.16e\033[0;0m".format(
            values[1], values[2], values[3])
    print(formatter % (ite, v, g_norm, d_norm), end="")
    return (ite, v, g_norm, d_norm)
