def read_events_one_hot(event_embedding_path):
    events={}
    with open(event_embedding_path, "r",encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_old = line.split(" ->", 1)[0].replace("\n", "")
            line_new = line.split(" ->", 1)[1].replace("\n", "").replace("[", "").replace("]", "").replace(" ", "")
            event_one_hot = [float(x) for x in line_new.split(",") if x]
            if line_old in events:
                continue
            else:
                events[line_old] = event_one_hot
    f.close()
    return events
def read_file_one_hot(file_embedding_path,dimension):

    file_name_map = {}
    with open(file_embedding_path,"r",encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            line_old = line.split(" ->", 1)[0].replace("\n", "")
            line_new = line.split(" ->", 1)[1].replace("\n", "").replace("[", "").replace("]", "").replace(" ", "")
            file_one_hot = [float(x) for x in line_new.split(",") if x] + [float(0) for i in range(96)]
            if line_old in file_name_map:
                continue
            else:
                file_name_map[line_old]=file_one_hot
    file.close()
    return file_name_map

def read_exception_one_hot(exception_embedding_path,dimension):
    exception_map = {}
    with open(exception_embedding_path,"r",encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            line_old = line.split(" ->", 1)[0].replace("\n", "")
            line_new = line.split(" ->", 1)[1].replace("\n", "").replace("[", "").replace("]", "").replace(" ", "")
            exception_one_hot = [float(x) for x in line_new.split(",") if x] + [float(0) for i in range(115)]
            if line_old in exception_map:
                continue
            else:
                exception_map[line_old]=exception_one_hot
    file.close()
    return exception_map