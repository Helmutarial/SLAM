"""
FrameInspector

Utility class to inspect and print attributes of SpectacularAI Frame, FrameSet, or KeyFrame objects.
"""

class FrameInspector:
    """
    Utility class to inspect and print attributes of SpectacularAI Frame, FrameSet, or KeyFrame objects.
    """
    @staticmethod
    def printAttributes(obj, objName="object"):
        """
        Print all attributes of an object (excluding dunder methods).
        Args:
            obj: Object to inspect
            objName: Name to display
        """
        print(f"\nAttributes of {objName}:")
        for attr in dir(obj):
            if not attr.startswith('__'):
                try:
                    value = getattr(obj, attr)
                    print(f"    {objName}.{attr}: {value}")
                except Exception as e:
                    print(f"    {objName}.{attr}: Error accessing ({e})")

    @staticmethod
    def inspectFrameSet(frameSet):
        """
        Print attributes of all frames in a FrameSet.
        Args:
            frameSet: FrameSet object
        """
        for frameAttr in ['primaryFrame', 'rgbFrame', 'depthFrame', 'secondaryFrame']:
            if hasattr(frameSet, frameAttr):
                frameObj = getattr(frameSet, frameAttr)
                FrameInspector.printAttributes(frameObj, frameAttr)
