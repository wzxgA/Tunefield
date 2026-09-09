export function buildModel(name) {
  return { name };
}

class Dataset {
  load(path) {
    this.path = path;
  }
}

