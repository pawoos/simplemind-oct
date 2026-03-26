from typing import Any

class SMCache:
    """
    SMCache is used to temporarily store data for datasets and samples.
    Examples:
        * incoming message data - in scenarios where messages may arrive out of order (e.g, when resize_tool is waiting for both an input image and a target image).
        * Result data - when results need to be aggregated over the dataset (e.g., when decision_tree_tool is aggragating training data).
    """

    def __init__(self):
        # message_cache stores messages by [dataset][sample][parameter_key]
        self.cache = {}

    def display(self):
        """
        Prints the current contents of the message cache.
        """
        
        print("---------- SMCache begin -----------")
        for dataset_key, sample_map in self.cache.items():
            for sample_key, parameter_map in sample_map.items():
                row_string = f"{dataset_key}/{sample_key}:"
                for parameter_key, value in parameter_map.items():
                    row_string += f" {parameter_key}: {value}"
                print(row_string)
        print("---------- SMCache end   -----------")

    def add(self, data: Any, sample_dict: dict, data_key: str):
        """
        Adds data to the cache under the appropriate dataset, sample, and data key.
        
        Args:
            data (Any): Data to be added to the cache.
            sample_dict (dict): Identifies the sample using a dictonary containing dataset, sample, and total.
            data_key (str): A key for this data within the sample cache.
                A parmeter name when it is being used to cache input parameters.
                A result name when storing results for a sample.      
        """
        dataset = sample_dict['dataset']
        sample = sample_dict['sample']
        self.cache \
            .setdefault(dataset, {}) \
            .setdefault(sample, {}) \
            .setdefault(data_key, data)

    def is_cached(self, sample_dict: dict, data_key: str) -> bool:
        """
        Checks if data for the given data_key in sample_dict is already cached.
        """
        dataset = sample_dict['dataset']
        sample = sample_dict['sample']
        return (
            dataset in self.cache and
            sample in self.cache[dataset] and
            data_key in self.cache[dataset][sample]
        )

    def get_dataset(self, dataset_id:str):
        """
        Retrieves a dataset from the cache. Unsafe if key doesn't exist.
        """
        return self.cache[dataset_id]
    
    def get_data(self, sample_dict, data_key: str):
        """
        Retrieves data for a sample from the cache. Unsafe if key doesn't exist.
        """
        return self.cache[sample_dict['dataset']][sample_dict['sample']][data_key]

    def get_sample(self, sample_dict):
        """
        Retrieves all messages for a sample. Returns {} if sample isn't cached.
        """
        dataset = sample_dict['dataset']
        sample = sample_dict['sample']
        return self.cache.get(dataset, {}).get(sample, {})

    def sample_inputs_complete(self, sample_dict, data_keys: list[str]) -> bool:
        """
        Checks if all specified (expected) data_keys have been received for a sample.
        """
        if data_keys is None:
            return True
        sample_cache = self.get_sample(sample_dict)
        return sample_cache is not None and set(sample_cache.keys()) == set(data_keys)

    def pop_sample(self, sample_dict):
        """
        Removes and returns all cached data for a sample.
        Deletes the dataset key if it becomes empty.
        """
        dataset = sample_dict['dataset']
        sample = sample_dict['sample']

        sample_cache = self.get_sample(sample_dict)
        if dataset in self.cache and sample in self.cache[dataset]:
            del self.cache[dataset][sample]
            if not self.cache[dataset]:  # remove dataset key if empty
                del self.cache[dataset]
        return sample_cache

    def all_samples_have_data(self, dataset_id:str, data_key: str, total_samples: int) -> bool: 
        """
        Returns True if all samples for this dataset have data (for the given data_key).
        """
        ds = self.cache.get(dataset_id)
        return (
            ds is not None
            and len(ds) >= total_samples
            and all(data_key in subdict for subdict in ds.values())
        )

    def del_dataset(self, dataset_id:str):
        """
        Deletes a dataset from the cache. Unsafe if key doesn't exist.
        """
        del self.cache[dataset_id]

def main():
    print("TESTING SMCache")

    mc = SMCache()
    sample_parameter = {'ik1': 'data1', 'ik2': 'data2'}
    n = 25
    sample_dict = {'dataset': 999, 'total': n}

    # Populate cache with complete samples
    for i in range(n - 5):
        sample_dict['sample'] = i
        for key in sample_parameter:
            mc.add(f"{sample_parameter[key]}_{i}", sample_dict, key)

    mc.display()

    # Remove and print sample 17
    sample_dict['sample'] = 17
    si = mc.pop_sample(sample_dict)
    print(si)
    mc.display()

    # Add incomplete data to remaining samples
    for i in range(n - 5, n):
        sample_dict['sample'] = i
        mc.add(f"{sample_parameter['ik1']}_{i}", sample_dict, 'ik1')

    mc.display()

    # Try popping a partially filled sample
    sample_dict['sample'] = 22
    si = mc.pop_sample(sample_dict)
    print(si)
    mc.display()


if __name__ == "__main__":
    main()
