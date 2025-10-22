class Validate:
    @staticmethod
    def is_uuid(value: str) -> bool:
        import re
        return re.match(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$', value) is not None
    
    @staticmethod
    def is_datetime(value: str) -> bool:
        try:
            import datetime
            if value.endswith('Z'):
                value = value[:-1]
            datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
            return True
        except ValueError:
            return False
        
    @staticmethod
    def is_float(value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def is_int(value: str) -> bool:
        try:
            int(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_valid_email(value: str) -> bool:
        import re
        return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value) is not None
        
    @staticmethod
    def is_valid_url(value: str) -> bool:
        import re
        return re.match(r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(:[0-9]+)?(/.*)?$', value) is not None
