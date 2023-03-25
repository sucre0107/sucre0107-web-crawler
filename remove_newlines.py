def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ',regex=False)
    serie = serie.str.replace('\r', ' ') #这行代码是为了去除\r，\r是回车符，\n是换行符,他们的区别是\r是回车符，\n是换行符，\r\n是回车换行符
    serie = serie.str.replace('  ', ' ')
    # serie = serie.str.replace('  ', ' ')
    return serie