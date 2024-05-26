def scuffed_translation(transcription):
    try:
        import re
        from word2number import w2n
        num_to_word_map = {
            '0': "zero",
            '1': "one",
            '2': "two",
            '3': "three",
            '4': "four",
            '5': "five",
            '6': "six",
            '7': "seven",
            '8': "eight",
            '9': "niner"
        }
        zero = ['zero']
        ones = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        teens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
        tens= ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
        hundreds = ['hundred']
        thousands = ['thousand']
        def pattern_join(number_pattern):
            out = r'\b|\b'.join(number_pattern)
            return r"\b"+out+r"\b"
        def pattern_extend(*args):
            base = []
            for i in args:
                base.extend(i)
            return base
        pattern_1 = pattern_join(pattern_extend(zero, ones)) #ones
        pattern_2 = pattern_join(teens) #teens
        pattern_3 = rf"({pattern_join(tens)}) ?({pattern_1})?" #tens
        pattern_4 = rf"({pattern_1}|{pattern_2}) ?({hundreds[0]}) ?(and)? ?({pattern_3}|{pattern_2}|{pattern_1})?" #hundreds
        pattern_5 = rf"({pattern_1}|{pattern_2}|{pattern_3}|{pattern_4}) ?{thousands[0]} ?(and)? ?({pattern_4}|{pattern_3}|{pattern_2}|{pattern_1})?"#thousands
        main_pattern = rf"({pattern_5})|({pattern_4})|({pattern_3})|({pattern_2})|({pattern_1})"
        regex = re.compile(main_pattern, re.IGNORECASE)
        best_match = [max(i, key=len) for i in re.findall(regex, transcription)]
        best_match = [i.strip(' ') for i in best_match]
        cp_transcribed = transcription
        for item in best_match:
            num = w2n.word_to_num(item)
            replacement = []
            for number in str(num):
                replacement.append(num_to_word_map[number])
            replacement = ' '.join(replacement)
            cp_transcribed = re.sub(rf"\b{item}\b", replacement, cp_transcribed)
        return cp_transcribed
    
    except Exception as e:
        print(e)
        return transcription